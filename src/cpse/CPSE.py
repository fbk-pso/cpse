# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of CPSE.
#
# CPSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CPSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#


import unified_planning as up
from ortools.sat.python import cp_model
from unified_planning.model import Effect, FNode, ProblemKind, timing
from unified_planning.model.scheduling import Activity, SchedulingProblem

from .CPSEBaseEngine import CPSEBaseEngine


class CPSE(CPSEBaseEngine):
    """
    Implementation of the CPSE Engine.

    This class extends the `CPSEBaseEngine` and provides concrete implementations
    for the abstract methods defined in the base class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "CPSE"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = CPSEBaseEngine.supported_kind()
        supported_kind.set_scheduling("OPTIONAL_ACTIVITIES")
        supported_kind.set_scheduling("SCOPED_CONSTRAINTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return bool(problem_kind <= CPSE.supported_kind())

    def check_if_supported_problem(self, problem: "up.model.AbstractProblem"):
        """
        Checks if the given problem is a supported instance of `SchedulingProblem`.
        Raises `NotImplementedError` if unsupported.

        Args:
            problem (up.model.AbstractProblem): The problem instance to validate.

        Raises:
            NotImplementedError: If the problem is not supported.
        """

        if not isinstance(problem, SchedulingProblem):
            raise NotImplementedError(f"Problem of type {type(problem)} not supported.")
        problem: SchedulingProblem

        if not problem.discrete_time:
            raise NotImplementedError("Continuous time not supported.")

        num_parametric_fluent_exps = sum(
            1
            for _timing, eff, _activity in problem.all_effects()
            if self._fluent_exp_contains_parameters(eff.fluent)
        )
        for fnode, _scope in problem.all_scoped_constraints():
            num_parametric_fluent_exps += len(
                self.extract_all_parametric_fluent_exp_from_fnode(fnode)
            )
        for _time_interval, fnode, _activity in problem.all_conditions():
            num_parametric_fluent_exps += len(
                self.extract_all_parametric_fluent_exp_from_fnode(fnode)
            )

        if num_parametric_fluent_exps > 0:
            raise NotImplementedError("Fluents with parameters are not supported.")

    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        for fnode, scope in problem.all_scoped_constraints():
            bool_var = self.add_constraint(fnode)
            constraint_var = self.model.add_bool_and([bool_var])
            if len(scope) > 0:
                constraint_var.only_enforce_if(
                    [self.fnode_to_value_or_variable(fn) for fn in scope]
                )

    def add_effects(self, problem: SchedulingProblem):
        """
        Adds the effects of the given scheduling problem to the model.

        Adds a reservoir constraint for each fluent in the scheduling problem.
        Each fluent's value is constrained to remain within the range [0, C],
        where C is the maximum capacity of the fluent.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                effects to be applied to the model.
        """

        # map each fluent to its effects, adjusting values based on increase/decrease
        # effect types
        fluent_effects: dict[
            FNode,
            list[
                tuple[
                    timing.Timing,
                    int,
                    cp_model.IntVar | cp_model.NotBooleanVariable | bool,
                ]
            ],
        ] = {}
        for effect_timing, eff, activity in problem.all_effects():
            eff: Effect
            fluent_exp = eff.fluent

            if eff.value.is_int_constant():
                value = eff.value.int_constant_value()
            elif eff.value.is_bool_constant():
                value = 1 if eff.value.bool_constant_value() else 0
            else:
                raise NotImplementedError(
                    "Effect values must be constants of type boolean or integer."
                )
            if value == 0:  # if value is 0, the effect is ignored
                continue

            if fluent_exp not in fluent_effects:
                fluent_effects[fluent_exp] = []

            if eff.is_increase():
                pass
            elif eff.is_decrease():
                value = -value
            else:
                # assignment effects not supported
                raise NotImplementedError(f"Effect kind {eff.kind} not supported.")

            bool_var: cp_model.IntVar | cp_model.NotBooleanVariable | bool
            if eff.is_conditional():
                if activity is not None and activity.optional:
                    bool_var = self.add_constraint(
                        problem.environment.expression_manager.And(
                            activity.present, eff.condition
                        )
                    )
                else:
                    bool_var = self.add_constraint(eff.condition)
            else:
                if activity is not None and activity.optional:
                    presence_var = self.fnode_to_value_or_variable(activity.present)
                    assert isinstance(
                        presence_var, (cp_model.IntVar, cp_model.NotBooleanVariable)
                    )
                    bool_var = presence_var
                else:
                    bool_var = True

            fluent_effects[fluent_exp].append((effect_timing, value, bool_var))

        for fluent_exp in fluent_effects:
            lb, ub = self._fluent_bounds[fluent_exp.fluent().name]
            if fluent_exp not in self._fluent_initial_value:
                raise NotImplementedError(
                    f"Fluent '{fluent_exp}' must be initialized with a constant value "
                    "of type integer or boolean."
                )
            init_value_exp = self._fluent_initial_value[fluent_exp]
            init_value: int
            if init_value_exp.is_int_constant():
                init_value = init_value_exp.int_constant_value()
            elif init_value_exp.is_bool_constant():
                if not init_value_exp.bool_constant_value():
                    init_value = 0
                else:
                    init_value = 1
            else:
                raise NotImplementedError(
                    "Only integer and boolean constants are supported as initial "
                    "values for fluents."
                )

            times: list[cp_model.LinearExprT] = [0]
            values: list[int] = [init_value]
            actives: list[cp_model.IntVar | cp_model.NotBooleanVariable | bool] = [True]
            for effect_timing, value, active in fluent_effects[fluent_exp]:
                times.append(self._convert_timing_to_linear_expr(effect_timing))
                values.append(value)
                actives.append(active)

            if lb > 0:
                raise NotImplementedError(
                    "Fluent lower bound cannot be greater than 0."
                )

            self.model.add_reservoir_constraint_with_active(
                times, values, actives, lb, ub
            )

    def add_condition(
        self,
        time_interval: timing.TimeInterval,
        fnode: FNode,
        activity: Activity | None,
        name: str,
    ):
        """
        Adds a condition to the model, enforcing that it is satisfied within a
        specified time interval.

        This method creates a constraint based on the given condition and uses the
        `add_cumulative`
        method to ensure it holds throughout the specified time interval.

        Args:
            time_interval (timing.TimeInterval): The time interval during which the
                condition
                must be satisfied.
            fnode (FNode): The FNode representing the condition to be added as a
                constraint.
            name (str): The name of the condition for identification within the model.
        """

        bool_var = self.add_constraint(fnode)

        start_delay = 0
        if time_interval.lower.delay != 0:
            assert isinstance(time_interval.lower.delay, int)
            start_delay += time_interval.lower.delay
        if time_interval.is_left_open():
            start_delay += 1
        start = self._model_vars[time_interval.lower.timepoint] + start_delay

        # add 1 to end because `add_cumulative` enforces the constraint for t in [start,
        # end),
        # but we want the constraint to be enforced also at the end
        end_delay = 1
        if time_interval.upper.delay != 0:
            assert isinstance(time_interval.upper.delay, int)
            end_delay += time_interval.upper.delay
        if time_interval.is_right_open():
            end_delay -= 1
        end = self._model_vars[time_interval.upper.timepoint] + end_delay

        duration = self.model.new_int_var(
            self.lower_bound,
            self.upper_bound,
            f"{name}_duration",
        )
        if activity is not None and activity.optional:
            interval_var = self.model.new_optional_interval_var(
                start,
                duration,
                end,
                self._model_vars[activity.present.presence()],
                name,
            )
        else:
            interval_var = self.model.new_interval_var(start, duration, end, name)

        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        for i, (time_interval, fnode, activity) in enumerate(problem.all_conditions()):
            self.add_condition(time_interval, fnode, activity, f"condition{i}")
