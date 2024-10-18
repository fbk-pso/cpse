from unified_planning.shortcuts import *
from unified_planning.model.scheduling import SchedulingProblem
from unified_planning.plans.schedule import Schedule
from unified_planning.environment import get_environment


def main():
    problem = SchedulingProblem("test")

    operators = problem.add_resource("operators", capacity=2)
    replenish_operators = problem.add_resource("replenish_operators", capacity=2)
    machines = problem.add_resource(f"machines", capacity=3)
    boxes = problem.add_resource(f"boxes", capacity=1)

    bool_var = problem.add_variable(
        "bool_var", get_environment().type_manager.BoolType()
    )
    int_var = problem.add_variable("int_var", get_environment().type_manager.IntType())

    problem.add_constraint(Implies(bool_var, Equals(int_var, 1)))

    box_construction = problem.add_activity(
        f"box_construction",
        duration=4,
    )
    box_construction.uses(machines, amount=1)
    box_construction.uses(operators, amount=1)
    box_construction.add_decrease_effect(box_construction.start, boxes, 1)
    box_construction.add_increase_effect(box_construction.end, boxes, 1)

    box_filling = problem.add_activity(f"box_filling", duration=2)
    box_filling.uses(operators, amount=1)
    box_filling.add_increase_effect(box_filling.start, boxes, 1)
    box_filling.add_decrease_effect(box_filling.end, boxes, 1)
    box_filling.add_effect(box_filling.end, machines, 2)
    # problem.add_constraint(
    #     # Or(
    #     #     LE(box_construction.end, box_filling.start),
    #     #     LE(box_construction.start, box_filling.end),
    #     # )
    #     # LE(box_construction.end, box_filling.start)
    #     LE(Plus(box_construction.end, +1), box_filling.start),
    # )

    replenishment = problem.add_activity(
        f"replenishment",
        duration=5,
    )
    replenishment.uses(replenish_operators, amount=1)

    replenishment.add_parameter(
        "replenishment_param", get_environment().type_manager.BoolType()
    )
    problem.add_decrease_effect(replenishment.start, boxes, 1)
    problem.add_increase_effect(replenishment.end, boxes, 1)
    problem.add_effect(replenishment.end, operators, 2)

    # TODO: not supported
    # fluent field of add_increase_effect must be a Fluent or a FluentExp
    # problem.add_increase_effect(replenishment.end, int_var, 1)

    problem.add_condition(
        ClosedTimeInterval(Timing(0, box_filling.start), Timing(0, box_filling.end)),
        LT(1, int_var),
    )
    # TODO: not supported
    # problem.add_condition(
    #     ClosedTimeInterval(box_filling.start, box_filling.end), LT(1, boxes)
    # )
    # box_filling.add_condition(ClosedTimeInterval(box_filling.start, box_filling.end), LT(1, int_var))
    box_filling.add_condition(
        ClosedTimeInterval(Timing(0, box_filling.start), Timing(0, box_filling.end)),
        LT(1, int_var),
    )

    problem.add_quality_metric(MinimizeMakespan())

    print(problem)

    # with OneshotPlanner(problem_kind=problem.kind) as planner:
    with OneshotPlanner(
        name="cpse", params={"lower_bound": 0, "upper_bound": 100}
    ) as planner:
        res = planner.solve(problem)
        print(res)


if __name__ == "__main__":
    main()
