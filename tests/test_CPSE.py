from cpse import CPSE

from unified_planning.shortcuts import *
from unified_planning.model.scheduling import SchedulingProblem

from .CommonTests import CommonTests, problem


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
