import pytest
from typing import List
from abc import abstractmethod

from cpse import CPSE, CPSETimepoints

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


class CommonTests:

    @abstractmethod
    def engine_name(self) -> str:
        pass

    @abstractmethod
    def engine_class(self):
        pass

    def problem_solved_satisficing_or_optimally(
        self, problem: SchedulingProblem
    ) -> PlanGenerationResult:
        with OneshotPlanner(name=self.engine_name()) as planner:
            res = planner.solve(problem)
            print(res.log_messages)
            assert res.status in [
                PlanGenerationResultStatus.SOLVED_SATISFICING,
                PlanGenerationResultStatus.SOLVED_OPTIMALLY,
            ]
            assert res.plan is not None
            assert isinstance(res.plan, Schedule)
            return res
        raise Exception(f"{self.engine_name()} engine cannot be loaded")

    def problem_unsolvable(self, problem: SchedulingProblem) -> PlanGenerationResult:
        with OneshotPlanner(name=self.engine_name()) as planner:
            res = planner.solve(problem)
            print(res.log_messages)
            assert res.status in [
                PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
                PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
            ]
            assert res.plan is None
            return res
        raise Exception(f"{self.engine_name()} engine cannot be loaded")

    def problem_unsupported(self, problem: SchedulingProblem) -> PlanGenerationResult:
        with OneshotPlanner(name=self.engine_name()) as planner:
            res = planner.solve(problem)
            print(res.log_messages)
            assert res.status == PlanGenerationResultStatus.UNSUPPORTED_PROBLEM
            assert res.plan is None
            return res
        raise Exception(f"{self.engine_name()} engine cannot be loaded")

    def test_new_engine(self):
        planner = self.engine_class()(lower_bound=1, upper_bound=2)
        assert planner.lower_bound == 1
        assert planner.upper_bound == 2

        with OneshotPlanner(
            name=self.engine_name(), params={"lower_bound": 3, "upper_bound": 100}
        ) as planner:
            assert isinstance(planner, self.engine_class())
            assert planner.lower_bound == 3
            assert planner.upper_bound == 100

    def test_activity_uses_resource(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)
        activity = problem.add_activity("activity", duration=1)
        activity.uses(resource, 1)
        self.problem_solved_satisficing_or_optimally(problem)

    def test_activity_uses_more_resources(self, problem: SchedulingProblem):
        resource1 = problem.add_resource("resource1", capacity=1)
        resource2 = problem.add_resource("resource2", capacity=2)
        resource3 = problem.add_resource("resource3", capacity=3)

        activity = problem.add_activity("activity", duration=1)
        activity.uses(resource1, 1)
        activity.uses(resource2, 2)
        activity.uses(resource3, 3)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_activities_use_same_resource(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)

        activity1 = problem.add_activity("activity1", duration=1)
        activity1.uses(resource, 1)

        activity2 = problem.add_activity("activity2", duration=1)
        activity2.uses(resource, 1)

        activity3 = problem.add_activity("activity3", duration=1)
        activity3.uses(resource, 1)

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert not are_activities_overlapped(res.plan, res.plan.activities)

    def test_set_activity_duration_bounds(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)
        activity.set_duration_bounds(10, 14)
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() == 10
        assert res.plan.get(activity.end).constant_value() == 14

    def test_activity_duration_as_parameter_exp1(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(int_var, 5))
        activity.set_fixed_duration(Times(int_var, 2))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            res.plan.get(activity.end).constant_value()
            - res.plan.get(activity.start).constant_value()
        ) == 10

    def test_activity_duration_as_parameter_exp2(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(int_var, 5))
        activity.set_duration_bounds(Times(int_var, 2), Times(int_var, 3))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() == 10
        assert res.plan.get(activity.end).constant_value() == 15

    def test_activity_deadline_and_release_date(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)
        activity1.add_deadline(Plus(activity2.end, 3))
        activity1.add_release_date(Minus(activity2.start, 3))

        res = self.problem_solved_satisficing_or_optimally(problem)
        activity1_start = res.plan.get(activity1.start).constant_value()
        activity1_end = res.plan.get(activity1.end).constant_value()
        activity2_start = res.plan.get(activity2.start).constant_value()
        activity2_end = res.plan.get(activity2.end).constant_value()
        assert (activity2_start - 3) <= activity1_start
        assert activity1_end <= (activity2_end + 3)

    def test_set_fluent_initial_value(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        problem.set_initial_value(resource, 1)

        activity = problem.add_activity("activity", duration=1)
        activity.uses(resource, 2)

        self.problem_unsolvable(problem)

    def test_problem_variables(self, problem: SchedulingProblem):
        problem.add_activity("activity", duration=1)

        bool_var = problem.add_variable(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert isinstance(res.plan.get(bool_var).constant_value(), bool)
        assert res.plan.get(int_var).constant_value() >= 0

    def test_activity_parameters(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)

        bool_var = activity.add_parameter(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = activity.add_parameter(
            "int_var", get_environment().type_manager.IntType()
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert isinstance(res.plan.get(bool_var).constant_value(), bool)
        assert res.plan.get(int_var).constant_value() >= 0

    def test_bounded_int_parameters(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)

        int_var1 = problem.add_variable(
            "int_var1",
            get_environment().type_manager.IntType(lower_bound=10, upper_bound=11),
        )
        int_var2 = activity.add_parameter(
            "int_var2",
            get_environment().type_manager.IntType(lower_bound=20, upper_bound=21),
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert 10 <= res.plan.get(int_var1).constant_value() <= 11
        assert 20 <= res.plan.get(int_var2).constant_value() <= 21

    def test_problem_constraints(self, problem: SchedulingProblem):
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
        problem.add_constraint(
            Iff(Equals(activity1.end, activity2.start), activity1_first)
        )
        problem.add_constraint(Equals(activity1.end, activity2.start))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1_first).constant_value()
        assert (
            res.plan.get(activity1.end).constant_value()
            == res.plan.get(activity2.start).constant_value()
        )

    def test_activity_constraints(self, problem: SchedulingProblem):
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

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1_first).constant_value()
        assert (
            res.plan.get(activity1.end).constant_value()
            == res.plan.get(activity2.start).constant_value()
        )

    def test_problem_effects(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)
        activity = problem.add_activity("activity", duration=20)

        problem.add_decrease_effect(activity.start, resource, 1)
        problem.add_increase_effect(10, resource, 1)

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() <= 10

    def test_activity_effects(self, problem: SchedulingProblem):
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

        self.problem_solved_satisficing_or_optimally(problem)

    def test_multiple_effects_at_the_same_timepoint(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        activity = problem.add_activity("activity", duration=20)

        problem.add_decrease_effect(activity.start, resource, 2)
        problem.add_increase_effect(activity.start, resource, 1)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_conditional_effects(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=20)
        activity2 = problem.add_activity("activity2", duration=20)

        problem.add_increase_effect(
            activity1.start, resource, 1, condition=LE(10, activity1.start)
        )
        problem.add_decrease_effect(activity2.start, resource, 1)

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1.start).constant_value() >= 10

    def test_problem_conditions(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        bool_var = problem.add_variable(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            LE(5, int_var),
        )
        problem.add_condition(
            TimeInterval(
                lower=Timing(
                    delay=0,
                    timepoint=Timepoint(TimepointKind.GLOBAL_START, container=None),
                ),
                upper=Timing(
                    delay=0,
                    timepoint=Timepoint(TimepointKind.GLOBAL_END, container=None),
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
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert 5 <= res.plan.get(int_var).constant_value() <= 10
        assert res.plan.get(bool_var).constant_value()

    def test_activity_conditions(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        bool_var = problem.add_variable(
            "bool_var", get_environment().type_manager.BoolType()
        )
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

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
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert 5 <= res.plan.get(int_var).constant_value() <= 10
        assert res.plan.get(bool_var).constant_value()

    def test_minimize_makespan(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=1)
        duration = 10
        num_activities = 5
        activities = []
        for i in range(num_activities):
            activity = problem.add_activity(f"activity{i}", duration=duration)
            activity.uses(resource, 1)
            activities.append(activity)

        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        for activity in activities:
            assert res.plan.get(activity.end).constant_value() <= (
                num_activities * duration
            )

    def test_int_fluent_with_constant_values(self, problem: SchedulingProblem):
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=0),
            default_initial_value=0,
        )

        activity = problem.add_activity("activity", duration=5)
        activity.add_increase_effect(activity.start + 1, fluent, 1)
        activity.add_decrease_effect(activity.end, fluent, 1)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_fluent_set_initial_value(self, problem: SchedulingProblem):
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=0, upper_bound=1),
        )
        problem.set_initial_value(fluent, 1)

        activity = problem.add_activity("activity", duration=5)
        activity.add_increase_effect(activity.start + 1, fluent, 1)
        activity.add_decrease_effect(activity.end, fluent, 1)

        self.problem_unsolvable(problem)

    def test_bool_fluent_with_constant_value(self, problem: SchedulingProblem):
        fluent = problem.add_fluent("fluent", default_initial_value=True)
        problem.set_initial_value(fluent, False)
        activity = problem.add_activity("activity", duration=5)
        self.problem_solved_satisficing_or_optimally(problem)

    def test_fluent_with_args(self, problem: SchedulingProblem):
        user_type_parent = UserType("user_type_parent")
        user_type = UserType("user_type", user_type_parent)
        fluent = problem.add_fluent(
            "fluent",
            IntType(lower_bound=0, upper_bound=10),
            obj=user_type_parent,
            default_initial_value=1,
        )
        o1 = problem.add_object("o1", user_type_parent)
        o2 = problem.add_object("o2", user_type)

        activity = problem.add_activity("activity", 2)
        activity.add_increase_effect(activity.end, fluent(o1), 1)
        activity.add_decrease_effect(activity.end, fluent(o2), 1)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_simple_problem(self, problem: SchedulingProblem):
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

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1_before_activity2).constant_value()
        assert res.plan.get(activity2_before_activity3).constant_value()
        assert res.plan.get(activity3.end).constant_value() == 19
        assert not are_activities_overlapped(
            res.plan, [activity1, activity2, activity3]
        )

    def test_not_supported_parameter_type(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        problem.add_variable("param1", get_environment().type_manager.RealType())
        activity.add_parameter("param2", get_environment().type_manager.RealType())
        assert not self.engine_class().supports(problem.kind)

    def test_not_supported_quality_metrics(self, problem: SchedulingProblem):
        problem.add_activity("activity", duration=5)
        problem.add_quality_metric(MinimizeSequentialPlanLength())
        assert not self.engine_class().supports(problem.kind)


class TestCPSE(CommonTests):
    def engine_name(self):
        return "cpse"

    def engine_class(self):
        return CPSE

    def test_not_supported_assign_effect(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        resource = problem.add_resource("resource", 10)
        problem.add_effect(activity.start, resource, 5)

        self.problem_unsupported(problem)

    def test_not_supported_effect_value(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        variable = problem.add_variable(
            "variable", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(variable, 1))
        activity = problem.add_activity("activity", duration=20)
        activity.add_decrease_effect(activity.start, resource, variable)

        self.problem_unsupported(problem)

    def test_not_supported_fluent_exp(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", 10)
        problem.add_constraint(LT(1, resource))

        self.problem_unsupported(problem)

    def test_not_supported_condition_with_fluent_exp(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        resource = problem.add_resource("resource", 10)
        problem.add_constraint(LT(1, resource))
        problem.add_condition(
            ClosedTimeInterval(activity.start, activity.end), LT(1, resource)
        )

        self.problem_unsupported(problem)

    def test_not_supported_uninitialized_fluent(self, problem: SchedulingProblem):
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=-1, upper_bound=11),
        )
        activity = problem.add_activity("activity", 5)
        problem.add_increase_effect(activity.start, fluent, 1)
        self.problem_unsupported(problem)

    def test_not_supported_fluent_initial_value_as_fluent_exp(
        self, problem: SchedulingProblem
    ):
        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(lower_bound=0),
            default_initial_value=1,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(lower_bound=0),
            default_initial_value=Times(fluent1, 2),
        )

        activity = problem.add_activity("activity", duration=5)
        activity.add_decrease_effect(activity.start, fluent1, 1)
        activity.add_decrease_effect(activity.start, fluent2, 2)

        self.problem_unsupported(problem)

    def test_not_supported_fluent_with_parameters(self, problem: SchedulingProblem):
        user_type = UserType("user_type")
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)
        parameter = problem.add_variable("parameter", user_type)
        fluent = problem.add_fluent(
            "fluent", IntType(), obj=user_type, default_initial_value=0
        )
        problem.set_initial_value(fluent(parameter), 1)

        activity = problem.add_activity("activity", 2)
        activity.add_increase_effect(activity.start, fluent(parameter), 1)

        self.problem_unsupported(problem)


class TestCPSETimepoints(CommonTests):
    def engine_name(self):
        return "cpse-timepoints"

    def engine_class(self):
        return CPSETimepoints

    def test_activity_duration_as_fluent_exp1(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(),
            default_initial_value=5,
        )
        activity.set_fixed_duration(Times(fluent, 2))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            res.plan.get(activity.end).constant_value()
            - res.plan.get(activity.start).constant_value()
        ) == 10

    def test_activity_duration_as_fluent_exp2(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)
        fluent = problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(),
            default_initial_value=5,
        )
        activity.set_duration_bounds(Times(fluent, 2), Times(fluent, 3))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() == 10
        assert res.plan.get(activity.end).constant_value() == 15

    def test_constraint_on_resource(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=1)
        resource = problem.add_resource("resource", capacity=2)

        problem.add_constraint(LE(resource, 2))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_assign_effect(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=4)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=4)
        activity2 = problem.add_activity("activity2", duration=5)
        activity3 = problem.add_activity("activity3", duration=6)

        activity1.add_effect(activity1.end, resource, 2)

        activity2.add_decrease_effect(activity2.start, resource, 2)
        problem.add_increase_effect(activity2.end, resource, 4)

        activity3.add_decrease_effect(activity3.start, resource, 4)

        # should be activity1.end <= activity2.start <= activity2.end <= activity3.start
        problem.add_constraint(
            Not(
                And(
                    LE(activity1.end, activity2.start),
                    LE(activity2.end, activity3.start),
                )
            )
        )
        problem.add_constraint(LE(activity1.end, 100))
        problem.add_constraint(LE(activity2.end, 100))
        problem.add_constraint(LE(activity3.end, 100))

        self.problem_unsolvable(problem)

    def test_effect_with_time_expression(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=4)
        problem.set_initial_value(resource, 0)

        activity1 = problem.add_activity("activity1", duration=4)
        activity2 = problem.add_activity("activity2", duration=5)

        activity1.add_effect(Timing(5, activity1.start), resource, 2)
        activity2.uses(resource)

        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity2.start).constant_value() == (
            res.plan.get(activity1.start).constant_value() + 5 + 1
        )

    def test_effect_with_value_expression(self, problem: SchedulingProblem):
        variable = problem.add_variable(
            "variable", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(variable, 10))

        resource = problem.add_fluent(
            "resource",
            get_environment().type_manager.IntType(),
            default_initial_value=0,
        )
        problem.add_constraint(LE(0, resource))

        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)

        problem.add_effect(activity1.end, resource, Times(variable, 2))
        problem.add_decrease_effect(activity2.start, resource, 20)

        problem.add_constraint(LE(activity2.start, activity1.start))
        problem.add_constraint(LE(activity1.end, 11))
        problem.add_constraint(LE(activity2.end, 11))

        self.problem_unsolvable(problem)

    def test_effect_with_fluent_expression(self, problem: SchedulingProblem):
        resource1 = problem.add_fluent(
            "resource1",
            get_environment().type_manager.IntType(),
            default_initial_value=0,
        )
        resource2 = problem.add_fluent(
            "resource2",
            get_environment().type_manager.IntType(),
            default_initial_value=1,
        )

        activity1 = problem.add_activity("activity1", duration=5)
        activity2 = problem.add_activity("activity2", duration=5)
        problem.add_constraint(LE(activity1.end, activity2.start))

        problem.add_increase_effect(activity1.start, resource1, Times(resource2, 1))
        problem.add_effect(activity1.end, resource1, Times(resource2, 2))
        activity2.uses(resource1, 2)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_conditional_effects2(self, problem: SchedulingProblem):
        resource1 = problem.add_resource("resource1", capacity=2)
        resource2 = problem.add_resource("resource2", capacity=2)
        problem.set_initial_value(resource1, 0)
        problem.set_initial_value(resource2, 0)

        activity1 = problem.add_activity("activity1", duration=20)
        activity2 = problem.add_activity("activity2", duration=30)

        problem.add_increase_effect(
            activity1.start, resource1, 1, condition=Equals(activity1.start, 10)
        )
        problem.add_decrease_effect(activity2.start, resource1, 1)

        problem.add_effect(
            activity1.end, resource2, 2, condition=Equals(activity1.end, 30)
        )
        problem.add_decrease_effect(activity2.end, resource2, 2)

        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity1.start).constant_value() == 10

    def test_conditional_effects_with_fluent_expression(
        self, problem: SchedulingProblem
    ):
        fluent1 = problem.add_fluent("fluent1", IntType(), default_initial_value=1)
        fluent2 = problem.add_fluent(
            "fluent2", IntType(), default_initial_value=fluent1
        )

        activity = problem.add_activity("activity", duration=20)

        problem.add_constraint(GE(fluent1, 0))
        problem.add_decrease_effect(
            activity.start, fluent1, Times(fluent2, 2), condition=LT(activity.start, 10)
        )
        problem.add_increase_effect(
            activity.start, fluent1, Times(fluent2, 2), condition=GE(activity.start, 10)
        )
        problem.add_quality_metric(MinimizeMakespan())

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(activity.start).constant_value() == 10

    def test_condition_with_ClosedTimeInterval(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        resource = problem.add_resource("resource", capacity=2)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            Equals(int_var, 5),
        )

        activity.add_decrease_effect(activity.start, resource, 1)
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            Equals(resource, 1),
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(int_var).constant_value() == 5

    def test_condition_with_TimePointInterval(self, problem: SchedulingProblem):
        resource = problem.add_resource("resource", capacity=2)
        activity = problem.add_activity("activity", duration=10)
        activity.uses(resource, 1)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            Equals(int_var, 5),
        )
        activity.add_condition(
            TimePointInterval(Timing(0, activity.start)),
            Equals(resource, 1),
        )
        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(int_var).constant_value() == 5

    def test_condition_forcing_equal_timepoint_values(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=10)
        activity2 = problem.add_activity("activity2", duration=10)
        activity3 = problem.add_activity("activity3", duration=10)
        resource = problem.add_resource("resource", capacity=3)

        # activity1.start == activity2.start == activity3.start == 0
        problem.add_constraint(Equals(activity1.start, 0))
        problem.add_constraint(Equals(activity2.start, 0))
        problem.add_constraint(Equals(activity3.start, 0))

        activity1.add_decrease_effect(activity1.start, resource, 1)
        activity2.add_decrease_effect(activity2.start, resource, 1)
        activity3.add_decrease_effect(activity3.start, resource, 1)

        activity1.add_condition(
            ClosedTimeInterval(Timing(0, activity1.start), Timing(0, activity1.start)),
            Equals(resource, 0),
        )

        self.problem_solved_satisficing_or_optimally(problem)

    def test_condition_without_resources(self, problem: SchedulingProblem):
        activity1 = problem.add_activity("activity1", duration=10)
        activity2 = problem.add_activity("activity2", duration=10)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        problem.add_constraint(LT(activity1.end, activity2.start))

        # these two conditions act as constraints since no resources are involved
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity1.start), Timing(0, activity1.end)),
            Equals(int_var, 5),
        )
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity2.start), Timing(0, activity2.end)),
            Equals(int_var, 4),
        )
        self.problem_unsolvable(problem)

    def test_condition_with_invalid_interval(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=10)
        resource = problem.add_resource("resource", capacity=2)
        int_var = problem.add_variable(
            "int_var", get_environment().type_manager.IntType()
        )

        # this condition should not be applied since the interval is [activity.end, activity.start]
        problem.add_condition(
            ClosedTimeInterval(Timing(0, activity.end), Timing(0, activity.start)),
            Equals(int_var, 5),
        )
        problem.add_constraint(Equals(int_var, 4))

        # this condition should not be applied since the interval is [activity.end, activity.start]
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.end), Timing(0, activity.start)),
            Equals(resource, 1),
        )

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert res.plan.get(int_var).constant_value() == 4

    def test_int_fluents(self, problem: SchedulingProblem):
        busy = problem.add_fluent(
            "busy", get_environment().type_manager.IntType(), default_initial_value=1
        )
        problem.set_initial_value(busy, 0)

        activity1 = problem.add_activity("activity1", duration=5)
        activity1.add_condition(activity1.start, Equals(busy, 0))
        activity1.add_effect(activity1.start + 1, busy, 1)
        activity1.add_effect(activity1.end, busy, 0)

        activity2 = problem.add_activity("activity2", duration=5)
        activity2.add_condition(activity2.start, Equals(busy, 0))
        activity2.add_effect(activity2.start + 1, busy, 1)
        activity2.add_effect(activity2.end, busy, 0)

        problem.add_constraint(LE(activity2.end, activity1.start))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            res.plan.get(activity2.end).constant_value()
            <= res.plan.get(activity1.start).constant_value()
        )

    def test_bool_fluents(self, problem: SchedulingProblem):
        busy = problem.add_fluent("busy", default_initial_value=True)
        problem.set_initial_value(busy, False)

        activity1 = problem.add_activity("activity1", duration=5)
        activity1.add_condition(activity1.start, Not(busy))
        activity1.add_effect(activity1.start + 1, busy, True)
        activity1.add_effect(activity1.end, busy, False)

        activity2 = problem.add_activity("activity2", duration=5)
        activity2.add_condition(activity2.start, Not(busy))
        activity2.add_effect(activity2.start + 1, busy, True)
        activity2.add_effect(activity2.end, busy, False)

        problem.add_constraint(LE(activity2.end, activity1.start))

        res = self.problem_solved_satisficing_or_optimally(problem)
        assert (
            res.plan.get(activity2.end).constant_value()
            <= res.plan.get(activity1.start).constant_value()
        )

    def test_fluents_with_parameters(self, problem: SchedulingProblem):
        activity = problem.add_activity("activity", duration=5)

        user_type_parent = UserType("user_type_parent")
        user_type1 = UserType("user_type1", user_type_parent)
        user_type2 = UserType("user_type2", user_type_parent)
        user_type3 = UserType("user_type3")
        param1 = activity.add_parameter("param1", user_type1)
        param2 = activity.add_parameter("param2", user_type2)
        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(),
            obj1=user_type_parent,
            obj2=user_type2,
            default_initial_value=2,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(),
            obj=user_type1,
            default_initial_value=0,
        )
        fluent3 = problem.add_fluent(
            "fluent3", BoolType(), obj=user_type3, default_initial_value=False
        )
        o1 = problem.add_object("o1", user_type1)
        o2 = problem.add_object("o2", user_type1)
        o3 = problem.add_object("o3", user_type2)
        o4 = problem.add_object("o4", user_type2)
        o5 = problem.add_object("o5", user_type_parent)
        o6 = problem.add_object("o6", user_type3)

        activity.add_increase_effect(activity.end, fluent1(param1, param2), 2)
        activity.add_effect(activity.start, fluent1(param1, param2), 2)
        activity.add_increase_effect(activity.end, fluent2(o1), 1)
        activity.add_effect(activity.start, fluent3(o6), True)

        problem.add_constraint(Equals(param1, o1))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_set_fluent_non_constant_initial_value(self, problem: SchedulingProblem):
        variable = problem.add_variable(
            "variable", get_environment().type_manager.IntType()
        )
        problem.add_constraint(Equals(variable, 1))

        resource1 = problem.add_fluent(
            "resource1",
            get_environment().type_manager.IntType(lower_bound=-1, upper_bound=11),
            default_initial_value=variable,
        )
        problem.add_constraint(LE(0, resource1))

        resource2 = problem.add_fluent(
            "resource2",
            get_environment().type_manager.IntType(),
        )
        problem.set_initial_value(resource2, Times(variable, 2))
        problem.add_constraint(LE(0, resource2))

        activity = problem.add_activity("activity", duration=5)
        activity.add_decrease_effect(activity.start, resource1, 1)
        activity.add_decrease_effect(activity.start, resource2, 2)
        activity.add_constraint(Equals(activity.start, 0))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_uninitialized_fluent(self, problem: SchedulingProblem):
        problem.add_fluent(
            "fluent",
            get_environment().type_manager.IntType(lower_bound=-1, upper_bound=11),
        )
        problem.add_activity("activity", 5)
        self.problem_solved_satisficing_or_optimally(problem)

    def test_set_fluent_initial_value_as_fluent_exp(self, problem: SchedulingProblem):
        fluent1 = problem.add_fluent(
            "fluent1",
            IntType(lower_bound=0),
            default_initial_value=1,
        )
        fluent2 = problem.add_fluent(
            "fluent2",
            IntType(lower_bound=0),
            default_initial_value=Times(fluent1, 2),
        )
        user_type = UserType("user_type")
        fluent3 = problem.add_fluent(
            "fluent3", IntType(), obj=user_type, default_initial_value=Times(fluent2, 2)
        )
        o1 = problem.add_object("o1", user_type)

        activity = problem.add_activity("activity", duration=5)
        activity.add_decrease_effect(activity.start, fluent1, 1)
        activity.add_decrease_effect(activity.start, fluent2, 2)
        activity.add_decrease_effect(activity.start, fluent3(o1), 2)

        self.problem_solved_satisficing_or_optimally(problem)

    def test_constraint_fluents_with_args(self, problem: SchedulingProblem):
        user_type = UserType("user_type")
        fluent = problem.add_fluent(
            "fluent",
            IntType(lower_bound=0, upper_bound=10),
            obj=user_type,
        )
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)

        activity = problem.add_activity("activity", 2)
        activity.add_constraint(Equals(fluent(o1), 2))
        activity.add_constraint(Equals(fluent(o2), 3))

        self.problem_solved_satisficing_or_optimally(problem)

    def test_condition_fluents_with_args(self, problem: SchedulingProblem):
        user_type = UserType("user_type")
        fluent = problem.add_fluent(
            "fluent", IntType(lower_bound=0, upper_bound=10), obj=user_type
        )
        o1 = problem.add_object("o1", user_type)
        o2 = problem.add_object("o2", user_type)
        # problem.set_initial_value(fluent(o1), 2)
        # problem.set_initial_value(fluent(o2), 3)

        activity = problem.add_activity("activity", 10)
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.start), Timing(0, activity.end)),
            Equals(fluent(o1), 2),
        )
        activity.add_condition(
            ClosedTimeInterval(Timing(0, activity.end), Timing(0, activity.end)),
            Equals(fluent(o2), 3),
        )

        self.problem_solved_satisficing_or_optimally(problem)
