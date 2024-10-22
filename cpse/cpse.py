#!/usr/bin/env python3
"""Unified Planning Integration for OR-Tools CP-SAT Model"""
from typing import IO, Callable, Optional, List, Union, Any, Dict
import collections
import operator
import pprint

import unified_planning as up
import unified_planning.engines.mixins as mixins
from unified_planning.engines import (
    Engine,
    Credits,
    PlanGenerationResult,
    PlanGenerationResultStatus,
)
from unified_planning.model.scheduling import SchedulingProblem, Activity
from unified_planning.engines.mixins.oneshot_planner import OptimalityGuarantee
from unified_planning.plans import Schedule
from unified_planning.model.operators import OperatorKind
from unified_planning.model.metrics import (
    MinimizeActionCosts,
    MinimizeSequentialPlanLength,
    MinimizeMakespan,
    MinimizeExpressionOnFinalState,
    MaximizeExpressionOnFinalState,
    Oversubscription,
    TemporalOversubscription,
)
from unified_planning.model.effect import Effect
from unified_planning.model.parameter import Parameter
from unified_planning.model.fnode import FNode
from unified_planning.model import timing


from ortools.sat.python import cp_model


# TODO: complete
credits = {
    "name": "cpse",
    "author": "",
    "contact": "",
    "website": "",
    "license": "GPLv3",
    "short_description": "",
    "long_description": "",
}

_SUPPORTED_KIND = up.model.ProblemKind(
    {
        # PROBLEM_CLASS
        "ACTION_BASED",
        "HIERARCHICAL",
        "SCHEDULING",
        # "CONTINGENT", "ACTION_BASED_MULTI_AGENT", "TAMP",
        # PROBLEM_TYPE
        "SIMPLE_NUMERIC_PLANNING",
        # "GENERAL_NUMERIC_PLANNING",
        # TIME
        "CONTINUOUS_TIME",
        "DISCRETE_TIME",
        "INTERMEDIATE_CONDITIONS_AND_EFFECTS",
        "EXTERNAL_CONDITIONS_AND_EFFECTS",
        "TIMED_EFFECTS",
        "TIMED_GOALS",
        "DURATION_INEQUALITIES",
        "SELF_OVERLAPPING",
        # EXPRESSION_DURATION
        "STATIC_FLUENTS_IN_DURATIONS",
        "FLUENTS_IN_DURATIONS",
        "INT_TYPE_DURATIONS",
        # "REAL_TYPE_DURATIONS",
        # NUMBERS
        # "CONTINUOUS_NUMBERS",
        "DISCRETE_NUMBERS",
        "BOUNDED_TYPES",
        # CONDITIONS_KIND
        "NEGATIVE_CONDITIONS",
        "DISJUNCTIVE_CONDITIONS",
        "EQUALITIES",
        # "EXISTENTIAL_CONDITIONS",
        # "UNIVERSAL_CONDITIONS",
        # EFFECTS_KIND
        # "CONDITIONAL_EFFECTS",
        "INCREASE_EFFECTS",
        "DECREASE_EFFECTS",
        # "FORALL_EFFECTS",
        "STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS",
        "STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS",
        "STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS",
        "FLUENTS_IN_BOOLEAN_ASSIGNMENTS",
        "FLUENTS_IN_NUMERIC_ASSIGNMENTS",
        "FLUENTS_IN_OBJECT_ASSIGNMENTS",
        # TYPING
        "FLAT_TYPING",
        "HIERARCHICAL_TYPING",
        # FLUENTS_TYPE
        "NUMERIC_FLUENTS",
        "OBJECT_FLUENTS",
        "INT_FLUENTS",
        # "REAL_FLUENTS",
        # PARAMETERS
        # "BOOL_FLUENT_PARAMETERS",
        # "BOUNDED_INT_FLUENT_PARAMETERS",
        "BOOL_ACTION_PARAMETERS",
        "BOUNDED_INT_ACTION_PARAMETERS",
        "UNBOUNDED_INT_ACTION_PARAMETERS",
        # "REAL_ACTION_PARAMETERS",
        # QUALITY_METRICS
        "ACTIONS_COST",
        "FINAL_VALUE",
        "MAKESPAN",
        "PLAN_LENGTH",
        # "OVERSUBSCRIPTION",
        # "TEMPORAL_OVERSUBSCRIPTION",
        "INT_NUMBERS_IN_OVERSUBSCRIPTION",
        # "REAL_NUMBERS_IN_OVERSUBSCRIPTION",
        # ACTIONS_COST_KIND
        "STATIC_FLUENTS_IN_ACTIONS_COST",
        "FLUENTS_IN_ACTIONS_COST",
        "INT_NUMBERS_IN_ACTIONS_COST",
        # "REAL_NUMBERS_IN_ACTIONS_COST",
        # SIMULATED_ENTITIES
        # "SIMULATED_EFFECTS",
        # CONSTRAINTS_KIND
        # "TRAJECTORY_CONSTRAINTS",
        # "STATE_INVARIANTS"
        # HIERARCHICAL
        "METHOD_PRECONDITIONS",
        "TASK_NETWORK_CONSTRAINTS",
        "INITIAL_TASK_NETWORK_VARIABLES",
        "TASK_ORDER_TOTAL",
        "TASK_ORDER_PARTIAL",
        # "TASK_ORDER_TEMPORAL",
        # INITIAL_STATE
        "UNDEFINED_INITIAL_NUMERIC",
        "UNDEFINED_INITIAL_SYMBOLIC",
    },
    version=2,
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
    # OperatorKind.BOOL_CONSTANT: None,
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
    OperatorKind.AT_MOST_ONCE: None,
    # OperatorKind.DOT: None,
}

activity_type = collections.namedtuple(
    "activity_type", "start end interval up_activity"
)


class CPSE(
    Engine,
    mixins.OneshotPlannerMixin,
):
    def __init__(self, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        self.lower_bound = options.get("lower_bound", 0)
        self.upper_bound = options.get("upper_bound", cp_model.INT32_MAX)
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
        c = Credits(**credits)
        return c

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def supported_kind() -> up.model.ProblemKind:
        return _SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: up.model.ProblemKind) -> bool:
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
                raise NotImplementedError

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

        activities_effects = [
            (timing, eff)
            for act in problem.activities
            for timing, effects in act.effects.items()
            for eff in effects
        ]

        effect_timepoints = {}
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
                raise NotImplementedError

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
            # print("################")
            # print(fluent_name)
            # print(self.fluent_capacity[fluent_name])
            # print(times, values)
            self.model.add_reservoir_constraint(
                times, values, 0, self.fluent_capacity[fluent_name]
            )

        return effect_timepoints

    def add_constraint_rec(
        self, fnode: FNode, enforce_if=True
    ) -> Union[cp_model.IntVar, bool, int, None, Any]:
        """Recursively add the constraint (represented by the fnode) to the model"""

        if fnode.is_parameter_exp():
            return self.model_vars[fnode.parameter().name][0]

        elif fnode.is_timing_exp():
            # TODO: test
            timing = fnode.timing()
            var = self.model_vars[str(timing.timepoint)][0]
            # TODO: delay != 0 is possible?
            if timing.delay != 0:
                assert isinstance(timing.delay, int)
                return var + timing.delay
            return var

        elif fnode.is_constant():
            return fnode.constant_value()

        elif fnode.node_type in [
            OperatorKind.PLUS,
            OperatorKind.MINUS,
            OperatorKind.TIMES,
            OperatorKind.DIV,
        ]:
            op = _OPERATOR_MAP[fnode.node_type]
            args = [self.add_constraint_rec(arg) for arg in fnode.args]
            return op(*args)

        elif fnode.node_type == OperatorKind.NOT:
            return self.add_constraint_rec(fnode.args[0]).negated()

        elif fnode.node_type in [
            OperatorKind.AND,
            OperatorKind.OR,
            OperatorKind.IMPLIES,
            OperatorKind.IFF,
            OperatorKind.AT_MOST_ONCE,
        ]:
            bool_vars = []
            for arg in fnode.args:
                bool_vars.append(self.add_constraint_rec(arg))

            if fnode.node_type == OperatorKind.AND:
                constraint = self.model.add_bool_and(bool_vars)
            elif fnode.node_type == OperatorKind.OR:
                constraint = self.model.add_bool_or(bool_vars)
            elif fnode.node_type == OperatorKind.IMPLIES:
                assert len(bool_vars) == 2
                constraint = self.model.add_implication(bool_vars[0], bool_vars[1])
            elif fnode.node_type == OperatorKind.IFF:
                assert len(bool_vars) == 2
                # TODO: test
                constraint = self.model.add(bool_vars[0] == bool_vars[1])
            elif fnode.node_type == OperatorKind.AT_MOST_ONCE:
                # TODO: test
                constraint = self.model.add_at_most_one(bool_vars)

        elif fnode.node_type in [
            OperatorKind.LE,
            OperatorKind.LT,
            OperatorKind.EQUALS,
        ]:
            op = _OPERATOR_MAP[fnode.node_type]
            args = [self.add_constraint_rec(arg) for arg in fnode.args]
            constraint = self.model.add(op(*args))

        else:
            raise NotImplementedError(f"node type {fnode.node_type} not supported")

        if enforce_if:
            bool_var = self.new_bool_var()
            constraint.only_enforce_if(bool_var)
            return bool_var

    def add_constraints(self, problem: SchedulingProblem):
        """Add the problem constraints to the model"""

        for fnode in problem.base_constraints:
            self.add_constraint_rec(fnode, enforce_if=False)

        for act in problem.activities:
            for fnode in act.constraints:
                self.add_constraint_rec(fnode, enforce_if=False)

    def add_condition(
        self, time_interval: timing.TimeInterval, fnode: FNode, name: str
    ):
        """Add a condition to the model. Add a constraint and force it is satisfied
        in the time interval using the `add_cumulative` method."""

        bool_var = self.add_constraint_rec(fnode)

        # TODO: support time expressions
        start = self.model_vars[str(time_interval.lower.timepoint)][0]
        if time_interval.lower.delay != 0:
            assert isinstance(time_interval.lower.delay, int)
            start += time_interval.lower.delay
        if time_interval.is_left_open():
            start += 1

        end = self.model_vars[str(time_interval.upper.timepoint)][0]
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
        # self.model.add(duration == (end - start))  # TODO: constraint can be removed ?

        # TODO: reuse interval if present
        interval_var = self.model.new_interval_var(start, duration, end, name)
        self.model.add_cumulative([interval_var], [bool_var.negated()], 0)

    def add_conditions(self, problem: SchedulingProblem):
        """Add the conditions defined in the problem and the activities to the model"""

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

        # TODO: add support for all metrics
        for metric in problem.quality_metrics:
            if isinstance(metric, MinimizeActionCosts):
                raise NotImplementedError
            elif isinstance(metric, MinimizeSequentialPlanLength):
                raise NotImplementedError
            elif isinstance(metric, MinimizeMakespan):
                self.model.minimize(makespan_var)
            elif isinstance(metric, MinimizeExpressionOnFinalState):
                raise NotImplementedError
            elif isinstance(metric, MaximizeExpressionOnFinalState):
                raise NotImplementedError
            elif isinstance(metric, Oversubscription):
                raise NotImplementedError
            elif isinstance(metric, TemporalOversubscription):
                raise NotImplementedError

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

        effect_timepoints = self.add_effect_constraints(problem)

        # print("fluent_capacity", self.fluent_capacity)
        # pprint.pprint(effect_timepoints)
        # print("")

        self.add_constraints(problem)
        self.add_conditions(problem)

        self.add_quality_metrics(problem, makespan_var)

        solver = cp_model.CpSolver()
        status = solver.solve(self.model)

        # TODO: check metrics
        metrics = {
            "engine_internal_time": str(solver.wall_time),
            "wall_time": str(solver.wall_time),
            "user_time": str(solver.user_time),
            "objective_value": str(solver.objective_value),
            "best_objective_bound": str(solver.best_objective_bound),
            "branches": str(solver.num_branches),
            "conflicts": str(solver.num_conflicts),
            "num_booleans": str(solver.num_booleans),
        }

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # for name in self.model_vars:
            #     print(self.model_vars[name][0], solver.value(self.model_vars[name][0]))

            # print()
            # for fluent_name in effect_timepoints:
            #     for timing, value in effect_timepoints[fluent_name]:
            #         print(fluent_name, timing, value)

            # print()

            assignment = {}
            for cp_var, up_var in self.model_vars.values():
                assignment[up_var] = solver.value(cp_var)

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
