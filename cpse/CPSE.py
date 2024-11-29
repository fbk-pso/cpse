from typing import List, Tuple, Dict

import unified_planning as up
from unified_planning.model import (
    FNode,
    timing,
    ProblemKind,
    Effect,
)
from unified_planning.model.scheduling import SchedulingProblem

from .CPSEBaseEngine import CPSEBaseEngine
from ortools.sat.python import cp_model


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
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSE.supported_kind()

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

        parametric_fluent_exps = list(
            filter(
                lambda e: self._fluent_exp_contains_parameters(e[1].fluent),
                problem.all_effects(),
            )
        )
        for fnode, activity in problem.all_constraints():
            parametric_fluent_exps += list(
                self.extract_all_parametric_fluent_exp_from_fnode(fnode)
            )
        for time_interval, fnode, activity in problem.all_conditions():
            parametric_fluent_exps += list(
                self.extract_all_parametric_fluent_exp_from_fnode(fnode)
            )

        if len(parametric_fluent_exps) > 0:
            raise NotImplementedError("Fluents with parameters are not supported.")

    def add_constraints(self, problem: SchedulingProblem):
        """
        Adds all constraints from the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                constraints to be added to the model.
        """

        # TODO: avoid bool_var for the root node

        for fnode in problem.base_constraints:
            bool_var = self.add_constraint(fnode)
            self.model.add_bool_and([bool_var])

        for act in problem.activities:
            for fnode in act.constraints:
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])

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

        # map each fluent to its effects, adjusting values based on increase/decrease effect types
        fluent_effects: Dict[
            FNode, List[Tuple["timing.Timing", int, cp_model.IntVar]]
        ] = {}
        for timing, eff, activity in problem.all_effects():
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
                raise NotImplementedError(f"Effect kind {eff.kind} not supported.")

            if eff.is_conditional():
                bool_var = self.add_constraint(eff.condition)
                fluent_effects[fluent_exp].append((timing, value, bool_var))
            else:
                fluent_effects[fluent_exp].append((timing, value, True))

        for fluent_exp in fluent_effects:
            lb, ub = self._fluent_bounds[fluent_exp.fluent().name]
            if fluent_exp not in self._fluent_initial_value:
                raise NotImplementedError(
                    f"Fluent '{fluent_exp}' must be initialized with a constant value of type integer or boolean."
                )
            init_value = self._fluent_initial_value[fluent_exp]
            if init_value.is_int_constant():
                init_value = init_value.int_constant_value()
            elif init_value.is_bool_constant():
                if init_value.bool_constant_value() == False:
                    init_value = 0
                else:
                    init_value = 1
            else:
                raise NotImplementedError(
                    "Only integer and boolean constants are supported as initial values for fluents."
                )

            times = [0]
            values = [init_value]
            actives = [True]
            for timing, value, active in fluent_effects[fluent_exp]:
                times.append(self._convert_timing_to_linear_expr(timing))
                values.append(value)
                actives.append(active)

            if lb > 0:
                # TODO: add support for lb > 0
                raise NotImplementedError(
                    "Fluent lower bound cannot be greater than 0."
                )

            self.model.add_reservoir_constraint_with_active(
                times, values, actives, lb, ub
            )

    def add_condition(
        self, time_interval: timing.TimeInterval, fnode: FNode, name: str
    ):
        """
        Adds a condition to the model, enforcing that it is satisfied within a specified time interval.

        This method creates a constraint based on the given condition and uses the `add_cumulative`
        method to ensure it holds throughout the specified time interval.

        Args:
            time_interval (timing.TimeInterval): The time interval during which the condition
                must be satisfied.
            fnode (FNode): The FNode representing the condition to be added as a constraint.
            name (str): The name of the condition for identification within the model.
        """

        bool_var = self.add_constraint(fnode)

        start_delay = 0
        if time_interval.lower.delay != 0:
            start_delay += time_interval.lower.delay
        if time_interval.is_left_open():
            start_delay += 1
        start = self._model_vars[time_interval.lower.timepoint] + start_delay

        # add 1 to end because `add_cumulative` enforces the constraint for t in [start, end),
        # but we want the constraint to be enforced also at the end
        end_delay = 1
        if time_interval.upper.delay != 0:
            end_delay += time_interval.upper.delay
        if time_interval.is_right_open():
            end_delay -= 1
        end = self._model_vars[time_interval.upper.timepoint] + end_delay

        interval_key = f"interval [{time_interval.lower.timepoint} + {start_delay}, {time_interval.upper.timepoint} + {end_delay}]"
        if interval_key not in self._variables_cache:
            duration = self.model.new_int_var(
                self.lower_bound,
                self.upper_bound,
                f"{name}_duration",
            )
            interval_var = self.model.new_interval_var(start, duration, end, name)
            self._variables_cache[interval_key] = interval_var
        interval_var = self._variables_cache[interval_key]
        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """
        Adds the conditions of the given scheduling problem to the model.

        Args:
            problem (SchedulingProblem): The scheduling problem containing the
                conditions to be added to the model.
        """

        for i, (time_interval, fnode) in enumerate(problem.base_conditions):
            name = f"base_condition{i}"
            self.add_condition(time_interval, fnode, name)

        for act in problem.activities:
            i = 0
            for time_interval in act.conditions:
                for fnode in act.conditions[time_interval]:
                    name = f"{act.name}_condition{i}"
                    self.add_condition(time_interval, fnode, name)
                    i += 1
