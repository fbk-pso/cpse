from unified_planning.shortcuts import *
from unified_planning.model.scheduling import SchedulingProblem
from unified_planning.plans.schedule import Schedule
from unified_planning.engines import PlanGenerationResultStatus
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# resources:
# - 2 operators
# - 1 replenish_operators
# - 2 machines
# - 4 drawers for each machine
#
# activities:
# - box_construction
#   uses    machine
#           drawer
# - box_filling
#   uses    operators
# - replenishment
#   uses    drawer
#           replenish_operators
#
# constraints:
# - box_construction.end <= box_filling.start
# - prev_box_construction.end <= box_construction.start
# - prev_box_filling.end <= box_filling.start
# - prev_box_filling.end <= box_construction.end
# - box_construction_causing_replenish.end <= replenishment.start
# - replenishment.end <= box_construction.start
# - (r1.end + d <= r2.start) or (r2.end + d <= r1.start) for each replenishment r1, r2
#       where d is the distance + 1 between two machines


def plot_remaining_boxes(ax, x, dx, y, remaining_boxes):
    xi = x
    dxi = float(dx) / len(remaining_boxes)
    yi = y - 40
    for i, r in enumerate(remaining_boxes):
        ax.broken_barh(
            [[xi, dxi - 0.2]],
            (yi, max(r * 5, 1)),
            facecolors=("black"),
        )
        ax.text(
            x=xi - 0.1 + dxi / 2,
            y=yi - 5,
            s=str(r),
            ha="center",
            va="center",
            color="black",
        )
        xi += dxi


def plot_schedule(plan: Schedule, remaining_boxes) -> None:
    fig, ax = plt.subplots()
    # ax.set_ylim(0, 50)
    # ax.set_xlim(0, 160)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")

    ax.set_xticks(range(0, 200, 2))
    # ax.set_yticks([15, 25, 35])
    # ax.set_yticklabels(['1', '2', '3'])

    ax.grid(True)

    def start_time(activity):
        return plan.get(activity.start).constant_value()

    bars = [[] for _ in range(len(remaining_boxes))]
    colors = {
        "box_construction": "red",
        "box_filling": "green",
        "replenishment": "lightblue",
    }
    for activity in sorted(plan.activities, key=start_time):
        i = 0
        if "m1" in activity.name:
            i = 1
        if "box_construction" in activity.name:
            color = colors["box_construction"]
        elif "box_filling" in activity.name:
            color = colors["box_filling"]
        elif "replenishment" in activity.name:
            color = colors["replenishment"]

        start = plan.get(activity.start).constant_value()
        end = plan.get(activity.end).constant_value()
        bars[i].append(((start, end - start), activity, color))

    offset = 0
    for i in range(len(bars)):
        machine_activities_counter = 0
        for j, bar in enumerate(bars[i]):
            x, dx = bar[0]
            y, dy = -(j + offset) * 10, 8
            ax.broken_barh(
                [bar[0]],
                (y, dy),
                facecolors=(bar[2]),
            )
            ax.text(
                # x=x + dx + 0.2,
                x=x + dx / 2,
                y=y + dy / 2,
                s=bar[1].name.replace(f"_m{i}", ""),
                ha="center",
                va="center",
                color="black",
            )
            if bar[2] == colors["box_construction"]:
                plot_remaining_boxes(
                    ax, x, dx, y, remaining_boxes[i][machine_activities_counter + 1]
                )
                machine_activities_counter += 1
        offset += len(bars[i])

    custom_lines = [
        Line2D([0], [0], color=colors["box_construction"], lw=4),
        Line2D([0], [0], color=colors["box_filling"], lw=4),
        Line2D([0], [0], color=colors["replenishment"], lw=4),
    ]
    plt.legend(custom_lines, ["box_construction", "box_filling", "replenishment"])
    plt.show()


def next_drawer(drawers, actual_drawer, remaining_boxes):
    if remaining_boxes[drawers[actual_drawer]] > 0:
        return actual_drawer
    else:
        return (actual_drawer + 1) % len(drawers)


def main():
    num_operators = 2
    num_replenish_operators = 1
    num_machines = 2
    num_drawers = [4] * num_machines
    num_boxes = [[2] * num_drawers[0]] * num_machines
    # size_boxes = [[i for i in range(num_drawers[0])]] * num_machines
    # box2drawers = [dict([(i, [i]) for i in range(num_drawers[0])])] * num_machines
    box2drawers = [dict([(0, [0, 1]), (1, [2]), (2, [3])])] * num_machines
    replenish_duration = 2
    replenish_capacity = 2
    construction_buffer_size = 4

    # num_boxes[0][0] = 0

    max_size = len(box2drawers[0].keys())
    num_orders = 10
    box_construction_duration = 2
    orders = [
        [  # id, box size, duration
            (i, i % max_size, i % max_size + 1) for i in range(num_orders)
        ],
        [
            (i, max_size - (i % max_size) - 1, max_size - (i % max_size))
            for i in range(num_orders)
        ],
    ]
    # if num_machines > len(orders):
    #     orders += [orders[0]] * (num_machines - len(orders))

    problem = SchedulingProblem("IMA")

    operators = problem.add_resource("operators", capacity=num_operators)
    replenish_operators = problem.add_resource(
        "replenish_operators", capacity=num_replenish_operators
    )

    # TODO: machines and drawers resources are useless?
    machines = []
    drawers = []
    for i in range(num_machines):
        machines.append(problem.add_resource(f"machine{i}", capacity=1))
        drawers.append([])
        for d in range(num_drawers[i]):
            drawers[i].append(problem.add_resource(f"drawers{d}_m{i}", capacity=1))

    replenishments = []
    remaining_boxes_saved = []
    for i in range(num_machines):
        prev_order = None
        remaining_boxes = list(num_boxes[i])
        replenish_counter = 0
        activities_causing_replenish = {}
        replenishments.append([])
        actual_drawers = {}
        for k in box2drawers[i].keys():
            actual_drawers[k] = 0

        remaining_boxes_saved.append([list(remaining_boxes)])
        construction_buffer = []
        for idd, box_size, duration in orders[i]:
            drawer_idx = next_drawer(
                box2drawers[i][box_size], actual_drawers[box_size], remaining_boxes
            )
            actual_drawers[box_size] = drawer_idx
            drawer = box2drawers[i][box_size][drawer_idx]

            box_construction = problem.add_activity(
                f"box_construction{idd}_m{i}_box{box_size}",
                duration=box_construction_duration,
            )
            box_construction.uses(machines[i], amount=1)
            box_construction.uses(drawers[i][drawer], amount=1)

            box_filling = problem.add_activity(
                f"box_filling{idd}_m{i}_box{box_size}", duration=duration
            )
            box_filling.uses(operators, amount=1)

            construction_buffer.append((box_construction.end, box_filling.end))

            problem.add_constraint(LE(box_construction.end, box_filling.start))
            if prev_order is not None:
                prev_box_construction, prev_box_filling = prev_order
                problem.add_constraint(
                    LE(prev_box_construction.end, box_construction.start)
                )
                problem.add_constraint(LE(prev_box_filling.end, box_filling.start))
                # problem.add_constraint(
                #     LE(prev_box_filling.end, box_construction.end)
                # )

            if remaining_boxes[drawer] == 0:
                replenishment = problem.add_activity(
                    f"replenishment{replenish_counter}_m{i}_drawer{drawer}",
                    duration=replenish_duration,
                )
                replenishments[i].append(replenishment)
                replenishment.uses(drawers[i][drawer], amount=1)
                replenishment.uses(replenish_operators, amount=1)
                problem.add_constraint(LE(replenishment.end, box_construction.start))
                if drawer in activities_causing_replenish:
                    problem.add_constraint(
                        LE(
                            activities_causing_replenish[drawer].end,
                            replenishment.start,
                        )
                    )
                replenish_counter += 1
                remaining_boxes[drawer] = replenish_capacity

            remaining_boxes[drawer] -= 1
            if remaining_boxes[drawer] == 0:
                activities_causing_replenish[drawer] = box_construction

            remaining_boxes_saved[i].append(list(remaining_boxes))
            prev_order = (box_construction, box_filling)

        # construction buffer constraints
        for j in range(len(construction_buffer) - construction_buffer_size):
            start1, end1 = construction_buffer[j]
            start2, end2 = construction_buffer[j + construction_buffer_size]
            problem.add_constraint(LE(end1, start2))

    for i in range(len(replenishments)):
        for r1, replenish1 in enumerate(replenishments[i]):
            for j in range(i, len(replenishments)):
                for r2, replenish2 in enumerate(replenishments[j]):
                    if i == j and r2 <= r1:
                        continue

                    problem.add_constraint(
                        And(
                            Or(
                                LE(
                                    Plus(replenish1.end, j - i + 1),
                                    replenish2.start,
                                ),
                                LE(
                                    Plus(replenish2.end, j - i + 1),
                                    replenish1.start,
                                ),
                            ),
                            Iff(
                                LT(
                                    Plus(replenish1.start, j - i + 1),
                                    replenish2.start,
                                ),
                                Not(
                                    Equals(
                                        Plus(replenish2.end, j - i + 1),
                                        replenish1.end,
                                    )
                                ),
                            ),
                        )
                    )

                    problem.add_constraint(
                        LT(Plus(replenish1.start, Times(10, 3)), 1000)
                    )

    # replenishment = replenishments[0][0]
    # problem.add_condition(
    #     TimeInterval(
    #         lower=replenishment.start,
    #         upper=replenishment.end,
    #         is_left_open=False,
    #         is_right_open=False,
    #     ),
    #     LE(replenishment.start, replenishment.end),
    # )

    problem.add_quality_metric(MinimizeMakespan())
    # problem.add_quality_metric(MinimizeSequentialPlanLength())
    # TODO: minimize for each machine

    print(problem)
    # print(box2drawers)
    print(orders)
    # print(boxes)

    with OneshotPlanner(name="cpse", params={"upper_bound": 100}) as planner:
        # with OneshotPlanner(problem_kind=problem.kind) as planner:
        res = planner.solve(problem)
        print(res)
        if res.status in [
            PlanGenerationResultStatus.SOLVED_SATISFICING,
            PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        ]:
            plot_schedule(res.plan, remaining_boxes_saved)


if __name__ == "__main__":
    main()
