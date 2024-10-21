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
from unified_planning.model.timing import TimeInterval


from ortools.sat.python import cp_model


# TODO
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

    bool_var_counter = -1
    int_var_counter = -1

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

        start_var = self.model.new_int_var(
            self.lower_bound, self.upper_bound, "start_" + activity.name
        )
        end_var = self.model.new_int_var(
            self.lower_bound, self.upper_bound, "end_" + activity.name
        )
        duration = activity.duration.upper.constant_value()
        interval_var = self.model.new_interval_var(
            start_var, duration, end_var, activity.name
        )
        self.activities[activity.name] = activity_type(
            start=start_var, end=end_var, interval=interval_var, up_activity=activity
        )
        self.model_vars[str(activity.start)] = (start_var, activity.start)
        self.model_vars[str(activity.end)] = (end_var, activity.end)

    def add_effect_intervals(
        self, problem: SchedulingProblem, makespan_var
    ) -> Dict[str, List]:
        """Process all the effects (activity effects and problem effects) and add
        their support intervals to the model.

        If an activity has two effects on the same fluent — a decrease effect
        followed by an increase effect — the support interval is not created.
        Instead, the activity interval is used.

        Also updates `fluent_to_capacity` when an increase effect is present.
        """

        # TODO: Assuming only one effect (increase or decrease) at the start and/or
        # end of an activity. If multiple effects occur at the same timepoint, they
        # can be simplified to a single effect.

        effect_intervals: Dict[str, List] = {}
        for act in problem.activities:
            activity = self.activities[act.name]
            action_effects = {}
            for timing, effects in act.effects.items():
                start_or_end = "start" if timing.is_from_start() else "end"
                for eff in effects:
                    eff: Effect
                    fluent_name = eff.fluent.fluent().name
                    value = eff.value.constant_value()
                    assert value > 0

                    if eff.is_increase():
                        pass
                    elif eff.is_decrease():
                        value = -value
                    else:
                        raise NotImplementedError

                    if fluent_name in action_effects:
                        # TODO: remove assumption
                        assert start_or_end not in action_effects[fluent_name]
                        action_effects[fluent_name][start_or_end] = value
                    else:
                        action_effects[fluent_name] = {start_or_end: value}

            for fluent_name in action_effects:
                if fluent_name not in effect_intervals:
                    effect_intervals[fluent_name] = []

                if (
                    "start" in action_effects[fluent_name]
                    and "end" in action_effects[fluent_name]
                    and action_effects[fluent_name]["start"] < 0
                    and action_effects[fluent_name]["start"]
                    == -action_effects[fluent_name]["end"]
                ):
                    # use a resource
                    effect_intervals[fluent_name].append(
                        (activity.interval, action_effects[fluent_name]["end"])
                    )
                    print(
                        f"{act.name} uses",
                        fluent_name,
                        action_effects[fluent_name]["end"],
                    )
                    # TODO: when (decrease at start and increase at end) we can use the
                    # activity interval
                else:
                    for start_or_end, value in action_effects[fluent_name].items():
                        print(f"{act.name} {fluent_name} at {start_or_end} += {value}")

                        activity_start_or_end = (
                            activity.start if start_or_end == "start" else activity.end
                        )
                        if value > 0:
                            interval_name = (
                                f"{act.name}_increase_{fluent_name}@{start_or_end}"
                            )
                            start = 0
                            end = activity_start_or_end
                            duration = activity_start_or_end
                            self.fluent_to_capacity[fluent_name] += value
                        else:
                            interval_name = (
                                f"{act.name}_decrease_{fluent_name}@{start_or_end}"
                            )
                            start = activity_start_or_end
                            end = makespan_var  # TODO: better to use upper bound?
                            duration = self.model.new_int_var(
                                self.lower_bound,
                                self.upper_bound,
                                f"{act.name}_{fluent_name}_{start_or_end}_duration",
                            )
                            self.model.add(
                                duration == (makespan_var - activity_start_or_end)
                            )

                        interval_var = self.model.new_interval_var(
                            start,
                            duration,
                            end,
                            interval_name,
                        )
                        effect_intervals[fluent_name].append((interval_var, abs(value)))

        for i, (timing, eff) in enumerate(problem.base_effects):
            assert str(timing) in self.model_vars
            start_or_end_var = self.model_vars[str(timing)][0]
            fluent_name = eff.fluent.fluent().name
            value = eff.value.constant_value()
            assert value > 0

            if eff.is_increase():
                interval_name = f"base_effect{i}_increase_{fluent_name}@{timing}"
                print(
                    f"base increase effect on {fluent_name} at {start_or_end_var} += {value}"
                )
                self.fluent_to_capacity[fluent_name] += value
                start = 0
                end = start_or_end_var
                duration = start_or_end_var
            elif eff.is_decrease():
                interval_name = f"base_effect{i}_decrease_{fluent_name}@{timing}"
                print(
                    f"base decrease effect on {fluent_name} at {start_or_end_var} += -{value}"
                )
                start = start_or_end_var
                end = makespan_var  # TODO: better to use upper bound?
                duration = self.model.new_int_var(
                    self.lower_bound,
                    self.upper_bound,
                    f"{interval_name}_duration",
                )
                self.model.add(duration == (makespan_var - activity_start_or_end))
            else:
                raise NotImplementedError

            interval_var = self.model.new_interval_var(
                start,
                duration,
                end,
                interval_name,
            )
            effect_intervals[fluent_name].append((interval_var, value))

        # add a cumulative constraint for each fluent
        for fluent_name in effect_intervals:
            intervals, demands = list(zip(*effect_intervals[fluent_name]))
            self.model.add_cumulative(
                intervals, demands, self.fluent_to_capacity[fluent_name]
            )

        return effect_intervals

    def add_constraint_rec(
        self, fnode: FNode
    ) -> Union[cp_model.IntVar, bool, int, Any]:
        """Recursively add the constraint (represented by the fnode) to the model"""

        if fnode.is_parameter_exp():
            return self.model_vars[fnode.parameter().name][0]

        elif fnode.is_timing_exp():
            return self.model_vars[str(fnode.timing().timepoint)][0]

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

        # elif fnode.node_type in [OperatorKind.TIMES, OperatorKind.DIV]:
        #     # TODO: int_var can be avoided if args are integers
        #     int_var = self.new_int_var()
        #     args = [self.add_constraint_rec(arg) for arg in fnode.args]
        #     if fnode.node_type == OperatorKind.TIMES:
        #         self.model.add_multiplication_equality(int_var, args)
        #     else:
        #         assert len(args) == 2
        #         # TODO: float division not supported
        #         self.model.add_division_equality(int_var, args[0], args[1])
        #     return int_var

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

            bool_var = self.new_bool_var()
            if fnode.node_type == OperatorKind.AND:
                self.model.add_bool_and(bool_vars).only_enforce_if(bool_var)
            elif fnode.node_type == OperatorKind.OR:
                self.model.add_bool_or(bool_vars).only_enforce_if(bool_var)
            elif fnode.node_type == OperatorKind.IMPLIES:
                assert len(bool_vars) == 2
                self.model.add_implication(bool_vars[0], bool_vars[1]).only_enforce_if(
                    bool_var
                )
            elif fnode.node_type == OperatorKind.IFF:
                assert len(bool_vars) == 2
                self.model.add_implication(bool_vars[0], bool_vars[1]).only_enforce_if(
                    bool_var
                )
                self.model.add_implication(bool_vars[1], bool_vars[0]).only_enforce_if(
                    bool_var
                )
            elif fnode.node_type == OperatorKind.AT_MOST_ONCE:
                # TODO: test
                self.model.add_at_most_one(bool_vars).only_enforce_if(bool_var)

            return bool_var

        elif fnode.node_type in [
            OperatorKind.LE,
            OperatorKind.LT,
            OperatorKind.EQUALS,
        ]:
            op = _OPERATOR_MAP[fnode.node_type]
            args = [self.add_constraint_rec(arg) for arg in fnode.args]
            bool_var = self.new_bool_var()
            self.model.add(op(*args)).only_enforce_if(bool_var)
            return bool_var

        else:
            raise NotImplementedError(f"node type {fnode.node_type} not supported")

    def add_constraints(self, problem: SchedulingProblem):
        """Add the problem constraints to the model"""

        for fnode in problem.base_constraints:
            bool_var = self.add_constraint_rec(fnode)
            # TODO: avoid bool var for the root node
            self.model.add_bool_and([bool_var])

    def add_condition(self, time_interval: TimeInterval, fnode: FNode, name: str):
        """Add a condition to the model"""

        bool_var = self.add_constraint_rec(fnode)
        # TODO: manage open intervals and fixed time intervals
        start = self.model_vars[str(time_interval.lower)][0]
        end = self.model_vars[str(time_interval.upper)][0]
        duration = self.model.new_int_var(
            self.lower_bound,
            self.upper_bound,
            f"{name}_duration",
        )
        self.model.add(duration == (end - start))
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

        # map each fluent to its maximum capacity
        self.fluent_to_capacity = {}
        for f in problem.fluents:
            self.fluent_to_capacity[f.name] = problem.fluents_defaults[
                f
            ].constant_value()

        effect_intervals = self.add_effect_intervals(problem, makespan_var)

        print("fluent_to_capacity", self.fluent_to_capacity)
        pprint.pprint(effect_intervals)
        print("")

        self.add_constraints(problem)
        self.add_conditions(problem)

        self.add_quality_metrics(problem, makespan_var)

        solver = cp_model.CpSolver()
        status = solver.solve(self.model)

        # Statistics.
        print("\nStatistics")
        print(f"  - conflicts: {solver.num_conflicts}")
        print(f"  - branches : {solver.num_branches}")
        print(f"  - wall time: {solver.wall_time}s")
        print("")

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(
                f"{'Optimal' if status == cp_model.OPTIMAL else 'Feasible'} solution found."
            )
            print(f"Objective: {solver.objective_value}")
            print()

            for name in self.model_vars:
                print(self.model_vars[name][0], solver.value(self.model_vars[name][0]))

            print()
            for fluent_name in effect_intervals:
                for interval_var, value in effect_intervals[fluent_name]:
                    print(
                        interval_var,
                        solver.value(interval_var.start_expr()),
                        solver.value(interval_var.end_expr()),
                    )

            print()

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
                metrics=None,
            )
        else:
            print("No solution found.")
            return PlanGenerationResult(
                PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
                plan=None,
                engine_name=self.name,
                log_messages=None,
                metrics=None,
            )
