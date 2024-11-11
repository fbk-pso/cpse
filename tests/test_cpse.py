import pytest
from typing import List

from cpse import CPSE

from unified_planning.shortcuts import *
from unified_planning.model.scheduling import SchedulingProblem, Activity
from unified_planning.engines import PlanGenerationResultStatus, PlanGenerationResult
from unified_planning.plans import Schedule


@pytest.fixture
def problem() -> SchedulingProblem:
    return SchedulingProblem("test")


def are_activities_overlapped(plan: Schedule, activities: List[Activity]) -> bool:
    def start_time(activity):
        return plan.get(activity.start).constant_value()

    sorted_activities = sorted(activities, key=start_time)
    for i in range(1, len(sorted_activities)):
        end_prev = plan.get(sorted_activities[i - 1].end).constant_value()
        start = plan.get(sorted_activities[i].start).constant_value()
        if end_prev > start:
            return True
    return False


def problem_solved_satisficing_or_optimally(
    problem: SchedulingProblem,
) -> PlanGenerationResult:
    with OneshotPlanner(name="cpse") as planner:
        res = planner.solve(problem)
        assert res.status in [
            PlanGenerationResultStatus.SOLVED_SATISFICING,
            PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        ]
        assert res.plan is not None
        assert isinstance(res.plan, Schedule)
        return res
    raise Exception("cpse engine cannot be loaded")


def problem_unsolvable(problem: SchedulingProblem) -> PlanGenerationResult:
    with OneshotPlanner(name="cpse") as planner:
        res = planner.solve(problem)
        assert res.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
        assert res.plan is None
        return res
    raise Exception("cpse engine cannot be loaded")


def test_new_cpse_engine():
    planner = CPSE(lower_bound=1, upper_bound=2)
    assert planner.lower_bound == 1
    assert planner.upper_bound == 2

    with OneshotPlanner(
        name="cpse", params={"lower_bound": 1, "upper_bound": 100}
    ) as planner:
        assert isinstance(planner, CPSE)
        assert planner.lower_bound == 1
        assert planner.upper_bound == 100


def test_activity_uses_resource(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=1)
    activity = problem.add_activity("activity", duration=1)
    activity.uses(resource, 1)
    problem_solved_satisficing_or_optimally(problem)


def test_activity_uses_more_resources(problem: SchedulingProblem):
    resource1 = problem.add_resource("resource1", capacity=1)
    resource2 = problem.add_resource("resource2", capacity=2)
    resource3 = problem.add_resource("resource3", capacity=3)

    activity = problem.add_activity("activity", duration=1)
    activity.uses(resource1, 1)
    activity.uses(resource2, 2)
    activity.uses(resource3, 3)

    problem_solved_satisficing_or_optimally(problem)


def test_activities_use_same_resource(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=1)

    activity1 = problem.add_activity("activity1", duration=1)
    activity1.uses(resource, 1)

    activity2 = problem.add_activity("activity2", duration=1)
    activity2.uses(resource, 1)

    activity3 = problem.add_activity("activity3", duration=1)
    activity3.uses(resource, 1)

    res = problem_solved_satisficing_or_optimally(problem)
    assert not are_activities_overlapped(res.plan, res.plan.activities)


def test_set_activity_duration_bounds(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=1)
    activity.set_duration_bounds(10, 14)
    res = problem_solved_satisficing_or_optimally(problem)
    assert res.plan.get(activity.start).constant_value() == 10
    assert res.plan.get(activity.end).constant_value() == 14


def test_initial_value(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=2)
    problem.set_initial_value(resource, 1)

    activity = problem.add_activity("activity", duration=1)
    activity.uses(resource, 2)

    problem_unsolvable(problem)


def test_problem_variables(problem: SchedulingProblem):
    problem.add_activity("activity", duration=1)

    bool_var = problem.add_variable(
        "bool_var", get_environment().type_manager.BoolType()
    )
    int_var = problem.add_variable("int_var", get_environment().type_manager.IntType())

    res = problem_solved_satisficing_or_optimally(problem)
    assert isinstance(res.plan.get(bool_var).constant_value(), bool)
    assert res.plan.get(int_var).constant_value() >= 0


def test_activity_parameters(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=1)

    bool_var = activity.add_parameter(
        "bool_var", get_environment().type_manager.BoolType()
    )
    int_var = activity.add_parameter(
        "int_var", get_environment().type_manager.IntType()
    )

    res = problem_solved_satisficing_or_optimally(problem)
    assert isinstance(res.plan.get(bool_var).constant_value(), bool)
    assert res.plan.get(int_var).constant_value() >= 0


def test_problem_constraints(problem: SchedulingProblem):
    activity1 = problem.add_activity("activity1", duration=1)
    activity2 = problem.add_activity("activity2", duration=1)
    activity1_first = problem.add_variable(
        "activity1_first", get_environment().type_manager.BoolType()
    )

    problem.add_constraint(
        Or(
            Equals(activity1.end, activity2.start),
            Equals(activity1.start, activity2.end),
        )
    )
    problem.add_constraint(
        And(
            Implies(Equals(activity1.end, activity2.start), activity1_first),
            Implies(Equals(activity1.start, activity2.end), Not(activity1_first)),
        )
    )
    problem.add_constraint(Iff(Equals(activity1.end, activity2.start), activity1_first))
    problem.add_constraint(Equals(activity1.end, activity2.start))

    res = problem_solved_satisficing_or_optimally(problem)
    assert res.plan.get(activity1_first).constant_value()
    assert (
        res.plan.get(activity1.end).constant_value()
        == res.plan.get(activity2.start).constant_value()
    )


def test_activity_constraints(problem: SchedulingProblem):
    activity1 = problem.add_activity("activity1", duration=1)
    activity2 = problem.add_activity("activity2", duration=1)
    activity1_first = problem.add_variable(
        "activity1_first", get_environment().type_manager.BoolType()
    )

    activity1.add_constraint(
        Or(
            Equals(activity1.end, activity2.start),
            Equals(activity1.start, activity2.end),
        )
    )
    activity2.add_constraint(
        And(
            Implies(Equals(activity1.end, activity2.start), activity1_first),
            Implies(Equals(activity1.start, activity2.end), Not(activity1_first)),
        )
    )
    activity1.add_constraint(
        Iff(Equals(activity1.end, activity2.start), activity1_first)
    )
    activity2.add_constraint(Equals(activity1.end, activity2.start))

    res = problem_solved_satisficing_or_optimally(problem)
    assert res.plan.get(activity1_first).constant_value()
    assert (
        res.plan.get(activity1.end).constant_value()
        == res.plan.get(activity2.start).constant_value()
    )


def test_problem_conditions(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=10)
    bool_var = problem.add_variable(
        "bool_var", get_environment().type_manager.BoolType()
    )
    int_var = problem.add_variable("int_var", get_environment().type_manager.IntType())

    problem.add_condition(
        ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
        LE(5, int_var),
    )
    problem.add_condition(
        TimeInterval(
            lower=Timing(
                delay=0, timepoint=Timepoint(TimepointKind.GLOBAL_START, container=None)
            ),
            upper=Timing(
                delay=0, timepoint=Timepoint(TimepointKind.GLOBAL_END, container=None)
            ),
            is_left_open=True,
            is_right_open=True,
        ),
        LE(int_var, 10),
    )
    problem.add_condition(
        TimeInterval(
            lower=Timing(1, activity.start),
            upper=Timing(2, activity.end),
            is_left_open=True,
            is_right_open=True,
        ),
        Implies(LE(int_var, 15), bool_var),
    )
    problem.add_condition(
        TimeInterval(
            lower=GlobalStartTiming(10),
            upper=GlobalStartTiming(15),
            is_left_open=True,
            is_right_open=False,
        ),
        bool_var,
    )
    res = problem_solved_satisficing_or_optimally(problem)
    assert 5 <= res.plan.get(int_var).constant_value() <= 10
    assert res.plan.get(bool_var).constant_value()


def test_activity_conditions(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=10)
    bool_var = problem.add_variable(
        "bool_var", get_environment().type_manager.BoolType()
    )
    int_var = problem.add_variable("int_var", get_environment().type_manager.IntType())

    activity.add_condition(
        TimeInterval(
            lower=Timing(delay=1, timepoint=activity.start),
            upper=Timing(delay=0, timepoint=activity.end),
            is_left_open=True,
            is_right_open=False,
        ),
        LE(int_var, 10),
    )
    activity.add_condition(
        TimePointInterval(Timing(3, activity.start)),
        LE(5, int_var),
    )
    activity.add_condition(
        OpenTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
        LE(5, int_var),
    )
    activity.add_condition(
        ClosedTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
        LE(5, int_var),
    )
    activity.add_condition(
        LeftOpenTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
        LE(5, int_var),
    )
    activity.add_condition(
        RightOpenTimeInterval(Timing(3, activity.start), Timing(-2, activity.end)),
        LE(5, int_var),
    )

    problem.add_constraint(Implies(And(LE(1, int_var), LE(int_var, 20)), bool_var))
    res = problem_solved_satisficing_or_optimally(problem)
    assert 5 <= res.plan.get(int_var).constant_value() <= 10
    assert res.plan.get(bool_var).constant_value()


def test_problem_effects(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=1)
    activity = problem.add_activity("activity", duration=20)

    problem.add_decrease_effect(activity.start, resource, 1)
    problem.add_increase_effect(10, resource, 1)

    res = problem_solved_satisficing_or_optimally(problem)
    assert res.plan.get(activity.start).constant_value() <= 10


def test_activity_effects(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=4)
    problem.set_initial_value(resource, 0)

    activity1 = problem.add_activity("activity1", duration=4)
    activity2 = problem.add_activity("activity2", duration=4)
    activity3 = problem.add_activity("activity3", duration=4)

    problem.add_increase_effect(10, resource, 2)
    activity1.add_decrease_effect(activity1.start, resource, 2)

    problem.add_increase_effect(20, resource, 3)
    activity2.add_decrease_effect(activity2.start, resource, 3)

    problem.add_increase_effect(30, resource, 4)
    activity3.add_decrease_effect(activity3.start, resource, 4)

    problem_solved_satisficing_or_optimally(problem)


def test_conditional_effects(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=2)
    problem.set_initial_value(resource, 0)

    activity1 = problem.add_activity("activity1", duration=20)
    activity2 = problem.add_activity("activity2", duration=20)

    problem.add_increase_effect(
        activity1.start, resource, 1, condition=LE(10, activity1.start)
    )
    problem.add_decrease_effect(activity2.start, resource, 1)

    res = problem_solved_satisficing_or_optimally(problem)
    assert res.plan.get(activity1.start).constant_value() >= 10


def test_minimize_makespan(problem: SchedulingProblem):
    resource = problem.add_resource("resource", capacity=1)
    duration = 10
    num_activities = 5
    activities = []
    for i in range(num_activities):
        activity = problem.add_activity(f"activity{i}", duration=duration)
        activity.uses(resource, 1)
        activities.append(activity)

    problem.add_quality_metric(MinimizeMakespan())

    res = problem_solved_satisficing_or_optimally(problem)
    for activity in activities:
        assert res.plan.get(activity.end).constant_value() <= (
            num_activities * duration
        )


def test_simple_problem(problem: SchedulingProblem):
    machine = problem.add_resource("machine", capacity=1)

    # Define activities with specific durations
    activity1 = problem.add_activity("activity1", duration=5)
    activity2 = problem.add_activity("activity2", duration=10)
    activity3 = problem.add_activity("activity3", duration=4)

    activity2.set_duration_bounds(5, 15)

    # Each activity requires the machine resource
    activity2.uses(machine, 1)
    activity3.uses(machine, 1)

    # activity1.uses(machine, 1)
    problem.add_decrease_effect(activity1.start, machine, 1)
    problem.add_increase_effect(activity1.end, machine, 1)

    problem.add_constraint(Equals(activity2.end, activity3.start))

    activity1_before_activity2 = problem.add_variable(
        "activity1_before_activity2", get_environment().type_manager.BoolType()
    )
    activity2_before_activity3 = problem.add_variable(
        "activity2_before_activity3", get_environment().type_manager.BoolType()
    )
    problem.add_constraint(
        Iff(LE(activity1.end, activity2.start), activity1_before_activity2)
    )
    problem.add_constraint(
        Iff(LE(activity2.end, activity3.start), activity2_before_activity3)
    )

    # minimizing makespan, activity1 should be the first activity
    problem.add_quality_metric(MinimizeMakespan())

    res = problem_solved_satisficing_or_optimally(problem)
    assert res.plan.get(activity1_before_activity2).constant_value()
    assert res.plan.get(activity2_before_activity3).constant_value()
    assert res.plan.get(activity3.end).constant_value() == 19
    assert not are_activities_overlapped(res.plan, [activity1, activity2, activity3])


def test_not_supported_parameter_type(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=5)
    problem.add_variable("param1", get_environment().type_manager.RealType())
    activity.add_parameter("param2", get_environment().type_manager.RealType())
    assert not CPSE().supports(problem.kind)


def test_not_supported_quality_metrics(problem: SchedulingProblem):
    problem.add_activity("activity", duration=5)
    problem.add_quality_metric(MinimizeSequentialPlanLength())
    assert not CPSE().supports(problem.kind)


def test_not_supported_effect_type(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=5)
    resource = problem.add_resource("resource", 10)
    problem.add_effect(activity.start, resource, 5)

    with pytest.raises(NotImplementedError):
        with OneshotPlanner(name="cpse") as planner:
            planner.solve(problem)


def test_not_supported_fluent_exp(problem: SchedulingProblem):
    resource = problem.add_resource("resource", 10)
    problem.add_constraint(LT(1, resource))

    with pytest.raises(NotImplementedError):
        with OneshotPlanner(name="cpse") as planner:
            planner.solve(problem)


def test_not_supported_codition_with_fluent_exp(problem: SchedulingProblem):
    activity = problem.add_activity("activity", duration=5)
    resource = problem.add_resource("resource", 10)
    problem.add_constraint(LT(1, resource))
    problem.add_condition(
        ClosedTimeInterval(activity.start, activity.end), LT(1, resource)
    )

    with pytest.raises(NotImplementedError):
        with OneshotPlanner(name="cpse") as planner:
            planner.solve(problem)
