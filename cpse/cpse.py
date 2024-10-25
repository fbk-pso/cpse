#!/usr/bin/env python3
"""Unified Planning Integration for OR-Tools CP-SAT Model"""
from typing import IO, Callable, Optional, List, Union, Any, Dict
import collections
import operator

import unified_planning as up
from unified_planning.engines import (
    Credits,
    PlanGenerationResult,
    PlanGenerationResultStatus,
)
from unified_planning.engines.mixins import OptimalityGuarantee
from unified_planning.model import OperatorKind, Parameter, FNode, timing, ProblemKind
from unified_planning.model.scheduling import SchedulingProblem, Activity
from unified_planning.model.metrics import MinimizeMakespan
from unified_planning.plans import Schedule

from ortools.sat.python import cp_model


# TODO: complete
credits = Credits(
    "CPSE",
    "FBK PSO Unit",
    "",
    "",
    "GPLv3",
    "",
    "",
)

_OPERATOR_MAP = {
    OperatorKind.AND: None,
    OperatorKind.OR: None,
    OperatorKind.NOT: None,
    OperatorKind.IMPLIES: None,
    OperatorKind.IFF: None,
    # OperatorKind.EXISTS: None,
    # OperatorKind.FORALL: None,
    # OperatorKind.FLUENT_EXP: None,
    OperatorKind.PARAM_EXP: None,
    # OperatorKind.VARIABLE_EXP: None,
    # OperatorKind.OBJECT_EXP: None,
    OperatorKind.TIMING_EXP: None,
    OperatorKind.BOOL_CONSTANT: None,
    OperatorKind.INT_CONSTANT: None,
    # OperatorKind.REAL_CONSTANT: None,
    OperatorKind.PLUS: operator.add,
    OperatorKind.MINUS: operator.sub,
    OperatorKind.TIMES: operator.mul,
    OperatorKind.DIV: operator.floordiv,
    OperatorKind.LE: operator.le,
    OperatorKind.LT: operator.lt,
    OperatorKind.EQUALS: operator.eq,
    # OperatorKind.ALWAYS: None,
    # OperatorKind.SOMETIME: None,
    # OperatorKind.SOMETIME_BEFORE: None,
    # OperatorKind.SOMETIME_AFTER: None,
    # OperatorKind.AT_MOST_ONCE: None,
    # OperatorKind.DOT: None,
}

activity_type = collections.namedtuple(
    "activity_type", "start end interval up_activity"
)


class CPSE(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """Implementation of the CPSE Engine."""

    def __init__(self, **kwargs):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        self.lower_bound = kwargs.get("lower_bound", 0)
        self.upper_bound = kwargs.get("upper_bound", cp_model.INT32_MAX)
        self.model = cp_model.CpModel()

        self.activities = {}
        self.model_vars = {}
        self.fluent_capacity = {}
        self.fluent_initial_value = {}

        self.bool_var_counter = -1
        self.int_var_counter = -1

    @property
    def name(self) -> str:
        return "CPSE"

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return credits

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("SCHEDULING")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("EXTERNAL_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("TIMED_EFFECTS")
        supported_kind.set_time("TIMED_GOALS")
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_expression_duration("INT_TYPE_DURATIONS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_quality_metrics("MAKESPAN")
        return supported_kind

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= CPSE.supported_kind()

    def new_bool_var(self):
        """Add a new anonymous boolean variable to the model"""
        self.bool_var_counter += 1
        return self.model.new_bool_var(f"bool{self.bool_var_counter}")

    def new_int_var(self):
        """Add a new anonymous integer variable to the model"""
        self.int_var_counter += 1
        return self.model.new_int_var(
            self.lower_bound, self.upper_bound, f"int{self.int_var_counter}"
        )

    def add_parameters(self, parameters: List[Parameter]):
        """Add the parameters to the model as boolean or integer variables"""

        for param in parameters:
            if param.type.is_bool_type():
                var = self.model.new_bool_var(param.name)
            elif param.type.is_int_type():
                var = self.model.new_int_var(
                    self.lower_bound, self.upper_bound, param.name
                )
            else:
                raise NotImplementedError(f"Parameter type {param.type} not supported.")

            # TODO: check if name duplicated
            self.model_vars[param.name] = (var, param)

    def add_activity(self, activity: Activity):
        """Add an activity to the model. Each activity is modeled using an interval."""

        # assume FixedDuration or ClosedDurationInterval
        assert (
            activity.duration.lower.is_int_constant()
            and activity.duration.upper.is_int_constant()
            and not activity.duration.is_left_open()
            and not activity.duration.is_right_open()
        )

        lower = activity.duration.lower.constant_value()
        upper = activity.duration.upper.constant_value()

        if lower == upper:
            # FixedDuration
            start_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "start_" + activity.name
            )
            end_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "end_" + activity.name
            )
            duration = upper
        else:
            # ClosedDurationInterval
            start_var = lower
            end_var = upper
            duration = upper - lower

        interval_var = self.model.new_interval_var(
            start_var, duration, end_var, activity.name
        )
        self.activities[activity.name] = activity_type(
            start=start_var, end=end_var, interval=interval_var, up_activity=activity
        )
        self.model_vars[str(activity.start)] = (start_var, activity.start)
        self.model_vars[str(activity.end)] = (end_var, activity.end)

    def add_effect_constraints(self, problem: SchedulingProblem) -> Dict[str, List]:
        """Add a reservoir constraint for each fluent. The value of the fluent
        is constrained to remain between [0, C] where C is its maximum capacity"""

        # TODO: model conditions on effects
        # TODO: simplify opposite effects at the same timepoint (e.g. increase and
        # decrease at the same time)

        activities_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
        ]

        # map each fluent to its effects
        effect_timepoints: Dict[str, List] = {}
        for timing, eff in activities_effects + problem.base_effects:
            assert str(timing.timepoint) in self.model_vars
            assert isinstance(timing.delay, int)

            fluent_name = eff.fluent.fluent().name
            value = eff.value.constant_value()
            if eff.is_increase():
                pass
            elif eff.is_decrease():
                value = -value
            else:
                raise NotImplementedError(f"Effect kind {eff.kind} not supported.")

            if fluent_name not in effect_timepoints:
                effect_timepoints[fluent_name] = []
            effect_timepoints[fluent_name].append(
                (self.model_vars[str(timing.timepoint)][0] + timing.delay, value)
            )

        # add a reservoir constraint for each fluent: the value of the fluent
        # should remain between [0, C] where C is its maximum capacity
        for fluent_name in effect_timepoints:
            times, values = list(zip(*effect_timepoints[fluent_name]))
            # initially (at time 0) the fluent is equal to its initial value
            times = [0] + list(times)
            values = [self.fluent_initial_value[fluent_name]] + list(values)
            self.model.add_reservoir_constraint(
                times, values, 0, self.fluent_capacity[fluent_name]
            )

        return effect_timepoints

    def add_constraint(self, fnode: FNode) -> Union[cp_model.IntVar, bool, int, Any]:
        """Add the constraint (represented by the fnode) to the model."""

        # TODO: reuse bool_var for the same constraint
        # TODO: transform to normal form?
        # TODO: use cache

        stack = [(fnode, False)]
        results = []

        while len(stack) > 0:
            fnode, processed = stack.pop()

            if fnode.is_parameter_exp():
                results.append(self.model_vars[fnode.parameter().name][0])

            elif fnode.is_timing_exp():
                timing = fnode.timing()
                var = self.model_vars[str(timing.timepoint)][0]
                if timing.delay != 0:
                    assert isinstance(timing.delay, int)
                    var += timing.delay
                results.append(var)

            elif fnode.node_type in [
                OperatorKind.BOOL_CONSTANT,
                OperatorKind.INT_CONSTANT,
            ]:
                results.append(fnode.constant_value())

            elif fnode.node_type in [
                OperatorKind.PLUS,
                OperatorKind.MINUS,
                OperatorKind.TIMES,
                OperatorKind.DIV,
            ]:
                if not processed:
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                else:
                    op = _OPERATOR_MAP[fnode.node_type]
                    args = [results.pop() for arg in fnode.args]
                    results.append(op(*args))

            elif fnode.node_type == OperatorKind.NOT:
                if not processed:
                    stack.append((fnode, True))
                    stack.append((fnode.args[0], False))
                else:
                    results.append(results.pop().negated())

            elif fnode.node_type in [
                OperatorKind.AND,
                OperatorKind.OR,
                OperatorKind.IMPLIES,
                OperatorKind.IFF,
            ]:
                if not processed:
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                    continue

                args = [results.pop() for arg in fnode.args]
                bool_var = self.new_bool_var()
                if fnode.node_type == OperatorKind.AND:
                    self.model.add_bool_and(args).only_enforce_if(bool_var)
                    self.model.add_bool_or([b.negated() for b in args]).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.OR:
                    self.model.add_bool_or(args).only_enforce_if(bool_var)
                    self.model.add_bool_and(
                        [b.negated() for b in args]
                    ).only_enforce_if(bool_var.negated())
                elif fnode.node_type == OperatorKind.IMPLIES:
                    assert len(args) == 2
                    self.model.add_implication(args[0], args[1]).only_enforce_if(
                        bool_var
                    )
                    self.model.add_bool_and(args[0], args[1].negated()).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.IFF:
                    assert len(args) == 2
                    self.model.add(args[0] == args[1]).only_enforce_if(bool_var)
                    self.model.add(args[0] != args[1]).only_enforce_if(
                        bool_var.negated()
                    )

                results.append(bool_var)

            elif fnode.node_type in [
                OperatorKind.LE,
                OperatorKind.LT,
                OperatorKind.EQUALS,
            ]:
                if not processed:
                    assert len(fnode.args) == 2
                    stack.append((fnode, True))
                    for arg in fnode.args:
                        stack.append((arg, False))
                    continue

                args = [results.pop() for arg in fnode.args]
                op = _OPERATOR_MAP[fnode.node_type]
                bool_var = self.new_bool_var()
                self.model.add(op(args[0], args[1])).only_enforce_if(bool_var)
                if fnode.node_type == OperatorKind.EQUALS:
                    self.model.add(args[0] != args[1]).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.LE:
                    self.model.add(args[0] > args[1]).only_enforce_if(
                        bool_var.negated()
                    )
                elif fnode.node_type == OperatorKind.LT:
                    self.model.add(args[0] >= args[1]).only_enforce_if(
                        bool_var.negated()
                    )

                results.append(bool_var)

            else:
                raise NotImplementedError(f"Node type {fnode.node_type} not supported.")

        assert len(results) == 1
        return results[0]

    def add_constraints(self, problem: SchedulingProblem):
        """Add the problem constraints to the model"""

        # TODO: avoid bool_var for the root node

        for fnode in problem.base_constraints:
            bool_var = self.add_constraint(fnode)
            self.model.add_bool_and([bool_var])

        for act in problem.activities:
            for fnode in act.constraints:
                bool_var = self.add_constraint(fnode)
                self.model.add_bool_and([bool_var])

    def add_condition(
        self, time_interval: timing.TimeInterval, fnode: FNode, name: str
    ):
        """Add a condition to the model. Add a constraint and force it is satisfied
        in the time interval using the `add_cumulative` method."""

        bool_var = self.add_constraint(fnode)

        start = self.model_vars[str(time_interval.lower.timepoint)][0]
        if time_interval.lower.delay != 0:
            assert isinstance(time_interval.lower.delay, int)
            start += time_interval.lower.delay
        if time_interval.is_left_open():
            start += 1

        # end + 1 because `add_cumulative` enforce the constraint for all t
        # st. start <= t < end
        end = self.model_vars[str(time_interval.upper.timepoint)][0] + 1
        if time_interval.upper.delay != 0:
            assert isinstance(time_interval.upper.delay, int)
            end += time_interval.upper.delay
        if time_interval.is_right_open():
            end -= 1

        duration = self.model.new_int_var(
            self.lower_bound,
            self.upper_bound,
            f"{name}_duration",
        )

        # TODO: reuse interval if present
        interval_var = self.model.new_interval_var(start, duration, end, name)
        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """Add the conditions defined in the problem and the activities to the model"""

        # TODO: conditions cannot be defined on fluents

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

    def add_quality_metrics(self, problem: SchedulingProblem, makespan_var):
        """Add the quality metrics to the model"""

        for metric in problem.quality_metrics:
            if isinstance(metric, MinimizeMakespan):
                self.model.minimize(makespan_var)
            else:
                raise NotImplementedError(f"Quality metric {metric} not supported.")

    def _solve(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.results.PlanGenerationResult":
        assert isinstance(problem, SchedulingProblem), "problem type not supported"

        self.model.name = problem.name

        # map each fluent to its maximum capacity and initial value
        for f in problem.fluents:
            self.fluent_capacity[f.name] = problem.fluents_defaults[f].constant_value()
            self.fluent_initial_value[f.name] = problem.fluents_defaults[
                f
            ].constant_value()

        # override initial values when an explicit one is defined
        for f in problem.explicit_initial_values:
            self.fluent_initial_value[f.fluent().name] = (
                problem.explicit_initial_values[f].constant_value()
            )

        # add the parameters of the problem to the model
        self.add_parameters(problem.base_variables)
        # add the parameters of the activities to the model
        for up_activity in problem.activities:
            self.add_parameters(up_activity.parameters)

        # add the activities to the model
        for up_activity in problem.activities:
            self.add_activity(up_activity)

        # define the makespan variable
        makespan_var = self.model.new_int_var(
            self.lower_bound, self.upper_bound, "makespan"
        )
        self.model.add_max_equality(
            makespan_var, [a.end for a in self.activities.values()]
        )

        # add global start and end to the model variables
        global_start = timing.GlobalStartTiming(delay=0)
        global_end = timing.GlobalEndTiming()
        self.model_vars[str(global_start.timepoint)] = (0, global_start)
        self.model_vars[str(global_end.timepoint)] = (makespan_var, global_end)

        # add the effects as constraints in the model
        self.add_effect_constraints(problem)

        # add problem-specific and activity-related constraints to the model
        self.add_constraints(problem)
        # add problem-specific and activity-related conditions to the model
        self.add_conditions(problem)

        # add quality metrics to the model
        self.add_quality_metrics(problem, makespan_var)

        # solve the modeled problem
        solver = cp_model.CpSolver()
        status = solver.solve(self.model)

        # define metrics to be returned with the result
        metrics = {
            "wall_time": str(solver.wall_time),
            "user_time": str(solver.user_time),
            "objective_value": str(solver.objective_value),
            "best_objective_bound": str(solver.best_objective_bound),
            "branches": str(solver.num_branches),
            "conflicts": str(solver.num_conflicts),
            "num_booleans": str(solver.num_booleans),
        }

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # map a decision variable to its solution value
            assignment = {}
            for cp_var, up_var in self.model_vars.values():
                assignment[up_var] = solver.value(cp_var)
                # map a boolean parameter to its boolean value rather than the
                # integer value returned by the solver
                if isinstance(up_var, Parameter) and up_var.type.is_bool_type():
                    assert solver.value(cp_var) in [0, 1]
                    assignment[up_var] = solver.value(cp_var) == 1

            plan = Schedule(problem.activities, assignment, problem.environment)

            return PlanGenerationResult(
                (
                    PlanGenerationResultStatus.SOLVED_OPTIMALLY
                    if status == cp_model.OPTIMAL
                    else PlanGenerationResultStatus.SOLVED_SATISFICING
                ),
                plan,
                engine_name=self.name,
                log_messages=None,
                metrics=metrics,
            )
        else:
            return PlanGenerationResult(
                PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
                plan=None,
                engine_name=self.name,
                log_messages=None,
                metrics=metrics,
            )
