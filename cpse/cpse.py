#!/usr/bin/env python3
"""Unified Planning Integration for OR-Tools CP-SAT Model"""
from typing import IO, Callable, Optional, List, Union, Any
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
from unified_planning.model.scheduling import SchedulingProblem
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
        self.bool_var_counter += 1
        return self.model.new_bool_var(f"bool{self.bool_var_counter}")

    def new_int_var(self):
        self.int_var_counter += 1
        return self.model.new_int_var(
            self.lower_bound, self.upper_bound, f"int{self.int_var_counter}"
        )

    def add_parameters(self, parameters: List[Parameter]):
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

    def add_constraint_rec(
        self, fnode: FNode
    ) -> Union[cp_model.IntVar, bool, int, Any]:
        if fnode.is_parameter_exp():
            return self.model_vars[fnode.parameter().name][0]

        elif fnode.is_timing_exp():
            # print(fnode.timing().timepoint)
            return self.model_vars[str(fnode.timing().timepoint)][0]

        elif fnode.is_constant():
            # print(fnode.constant_value())
            return fnode.constant_value()

        elif fnode.node_type in [
            OperatorKind.PLUS,
            OperatorKind.MINUS,
            OperatorKind.TIMES,
            OperatorKind.DIV,
        ]:
            op = _OPERATOR_MAP[fnode.node_type]
            # print(op)
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
            # print("not")
            # print(fnode.args)
            return self.add_constraint_rec(fnode.args[0]).negated()

        elif fnode.node_type in [
            OperatorKind.AND,
            OperatorKind.OR,
            OperatorKind.IMPLIES,
            OperatorKind.IFF,
            OperatorKind.AT_MOST_ONCE,
        ]:
            # print(fnode.node_type)
            # print(fnode.args)
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
            # print(op)
            args = [self.add_constraint_rec(arg) for arg in fnode.args]
            bool_var = self.new_bool_var()
            self.model.add(op(*args)).only_enforce_if(bool_var)
            # print(op(*args))
            return bool_var

        else:
            raise NotImplementedError(f"node type {fnode.node_type} not supported")

    def add_effect(self, effect, resources, activity_timepoints_assignment):
        fluent_name = effect.fluent.fluent().name
        value = effect.value.constant_value()
        assert value > 0

        if effect.is_decrease():
            value = -value

        for i, bool_var in enumerate(activity_timepoints_assignment):
            if effect.is_assignment():
                self.model.add(resources[fluent_name][i + 1] == value).only_enforce_if(
                    bool_var
                )
                print(f"{bool_var} => ({resources[fluent_name][i + 1]} == {value})")

            else:
                self.model.add(
                    resources[fluent_name][i + 1] == (resources[fluent_name][i] + value)
                ).only_enforce_if(bool_var)
                print(
                    f"{bool_var} => ({resources[fluent_name][i + 1]} == {(resources[fluent_name][i] + value)})"
                )

    def all_tpi_with_effects_on_fluent(
        self, tpi, fluent_name, problem, activity_timepoints_assignment
    ):
        for act in problem.activities:
            for timing, effects in act.effects.items():
                for eff in effects:
                    if eff.fluent.fluent().name == fluent_name:
                        yield activity_timepoints_assignment[str(timing)][tpi]

        for timing, eff in problem.base_effects:
            if eff.fluent.fluent().name == fluent_name:
                yield activity_timepoints_assignment[str(timing)][tpi]

    def add_condition(self, time_interval: TimeInterval, fnode: FNode, name: str):
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

    def _solve(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.results.PlanGenerationResult":
        assert isinstance(problem, SchedulingProblem), "problem type not supported"

        self.model.name = problem.name

        self.add_parameters(problem.base_variables)
        for act in problem.activities:
            self.add_parameters(act.parameters)
        print(self.model_vars)
        print("")

        activities = {}
        for act in problem.activities:
            suffix = act.name
            start_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "start_" + suffix
            )
            end_var = self.model.new_int_var(
                self.lower_bound, self.upper_bound, "end_" + suffix
            )
            duration = act.duration.upper.constant_value()
            interval_var = self.model.new_interval_var(
                start_var, duration, end_var, suffix
            )
            activity = activity_type(
                start=start_var, end=end_var, interval=interval_var, up_activity=act
            )
            activities[act.name] = activity
            self.model_vars[str(act.start)] = (start_var, act.start)
            self.model_vars[str(act.end)] = (end_var, act.end)

        print(self.model_vars)
        print("")

        makespan_var = self.model.new_int_var(
            self.lower_bound, self.upper_bound, "makespan"
        )
        self.model.add_max_equality(makespan_var, [a.end for a in activities.values()])

        timepoints = [
            self.model.new_int_var(self.lower_bound, self.upper_bound, f"timepoint{i}")
            for i in range(len(problem.activities) * 2)
        ]
        # TODO: we assume timepoints are all different
        for i in range(len(timepoints) - 1):
            self.model.add(timepoints[i] < timepoints[i + 1])
            print(f"{timepoints[i]} < {timepoints[i + 1]}")

        activity_timepoints = []
        for activity in activities.values():
            activity_timepoints.append((activity.up_activity.start, activity.start))
            activity_timepoints.append((activity.up_activity.end, activity.end))
        activity_timepoints_assignment = {}
        all_condition_vars = []  # TODO: remove
        for i in range(len(activity_timepoints)):
            up_tp_i, activity_timepoint_i = activity_timepoints[i]
            condition_vars = []
            for j in range(len(activity_timepoints)):
                if i == j:
                    continue

                up_tp_j, activity_timepoint_j = activity_timepoints[j]
                condition_bool_var = self.new_bool_var()
                condition_vars.append(condition_bool_var)
                all_condition_vars.append(condition_bool_var)
                # self.model.add(
                #     (activity_timepoint_i > activity_timepoint_j) == condition_bool_var
                # )
                self.model.add(
                    activity_timepoint_i > activity_timepoint_j
                ).only_enforce_if(condition_bool_var)
                print(
                    f"{condition_bool_var} => ({activity_timepoint_i} > {activity_timepoint_j})"
                )

            activity_timepoints_assignment[str(up_tp_i)] = []
            for idx, tp in enumerate(timepoints):
                # if tp is idx-th <=> (tp_i == activity_timepoint_i)
                # self.model.add((sum(condition_vars) == idx) == (tp == activity_timepoint_i))
                # self.model.add((sum(condition_vars) == idx) == bool_var)
                # self.model.add((tp == activity_timepoint_i) == bool_var)

                bool_var = self.model.new_bool_var(f"{activity_timepoint_i}@{tp}")
                activity_timepoints_assignment[str(up_tp_i)].append(bool_var)
                self.model.add(sum(condition_vars) == idx).only_enforce_if(bool_var)
                self.model.add(tp == activity_timepoint_i).only_enforce_if(bool_var)
                print(f"{bool_var} => ({sum(condition_vars)} == {idx})")
                print(f"{bool_var} => ({tp} == {activity_timepoint_i})")

            # each activity timepoint is assigned to exactly one timepoint
            print(f"exactly_one({activity_timepoints_assignment[str(up_tp_i)]})")
            self.model.add_exactly_one(activity_timepoints_assignment[str(up_tp_i)])

        # each timepoint is assigned to exactly one activity timepoint
        for i in range(len(timepoints)):
            self.model.add_exactly_one(
                bool_vars[i] for bool_vars in activity_timepoints_assignment.values()
            )

        # (num resources) x (num timepoints + 1)
        resources = {}
        for fluent in problem.fluents:
            init_value = self.model.new_constant(
                problem.fluents_defaults[fluent].constant_value()
            )
            resources[fluent.name] = [init_value]
            for i in range(len(timepoints)):
                resource_var = self.model.new_int_var(
                    0,
                    problem.fluents_defaults[fluent].constant_value(),
                    f"{fluent.name}@{timepoints[i]}",
                )
                resources[fluent.name].append(resource_var)

        # TODO: we assume maximum one effect (increase or decrease) at start and/or
        # end of each activity
        for act in problem.activities:
            activity = activities[act.name]
            for timing, effects in act.effects.items():
                for eff in effects:
                    self.add_effect(
                        eff,
                        resources,
                        activity_timepoints_assignment[str(timing)],
                    )

        for i, (timing, eff) in enumerate(problem.base_effects):
            assert str(timing) in self.model_vars
            self.add_effect(
                eff,
                resources,
                activity_timepoints_assignment[str(timing)],
            )

        # for each resource
        #   for each timepoint
        #       if no activities have effects on that resource at that timepoint
        #           resource@tp(i) == resource@tp(i-1)
        for fluent_name in resources:
            for i, tp in enumerate(timepoints):
                activity_tps = list(
                    self.all_tpi_with_effects_on_fluent(
                        i, fluent_name, problem, activity_timepoints_assignment
                    )
                )
                bool_var = self.new_bool_var()
                self.model.add(
                    resources[fluent_name][i + 1] == resources[fluent_name][i]
                ).only_enforce_if(bool_var)
                self.model.add_bool_or(activity_tps + [bool_var])
                print(
                    f"or({activity_tps}) or ({resources[fluent_name][i + 1]} == {resources[fluent_name][i]})"
                )

        for fnode in problem.base_constraints:
            bool_var = self.add_constraint_rec(fnode)
            # TODO: avoid bool var for the root node
            self.model.add_bool_and([bool_var])

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
            print("")

            for vv in activity_timepoints_assignment.values():
                for var in vv:
                    print(var, solver.value(var))
            for var in all_condition_vars:
                print(var, solver.value(var))
            for var in timepoints:
                print(var, solver.value(var))
            for _, var in activity_timepoints:
                print(var, solver.value(var))
            for name, rr in resources.items():
                print(name)
                for var in rr:
                    print(var, solver.value(var))

            print("")

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
