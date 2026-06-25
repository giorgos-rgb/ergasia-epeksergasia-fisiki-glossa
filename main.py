from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys


# ----------------------------
# 1) World model (Blocks World)
# ----------------------------


@dataclass(frozen=True)
class Obj:
    name: str
    color: str
    shape: str  # "cube", "pyramid", "box", "chest", "barrel", etc.
    is_container: bool = False
    capacity: int = 0  # How many items it can hold inside
    size: int = 1  # Physical size of the object itself


@dataclass
class World:
    objects: Dict[str, Obj]
    on: Dict[str, Optional[str]]  # on[x] = y means x is stacked on y
    inside: Dict[str, Optional[str]]  # inside[x] = y means x is inside container y
    open_containers: Dict[
        str, bool
    ]  # maps container name -> True (open) or False (closed)
    holding: Optional[str] = None

    def is_clear(self, obj_name: str) -> bool:
        """True if nothing is stacked on top of obj_name, and it's not trapped inside a closed container."""
        # Check if something is stacked on it
        if any(support == obj_name for support in self.on.values()):
            return False
        # Check if it's inside a closed container
        parent_container = self.inside.get(obj_name)
        if parent_container and not self.open_containers.get(parent_container, True):
            return False
        return True

    def top_of(self, obj_name: str) -> Optional[str]:
        """Return object that is on top of obj_name (if any)."""
        for x, support in self.on.items():
            if support == obj_name:
                return x
        return None

    def current_contents(self, container_name: str) -> List[str]:
        """Return a list of objects currently inside a container."""
        return [x for x, c in self.inside.items() if c == container_name]

    def describe(self) -> str:
        lines = []
        for x in sorted(self.objects):
            obj = self.objects[x]
            status = ""
            if obj.is_container:
                state = "open" if self.open_containers.get(x, True) else "closed"
                status = f" ({state}, capacity {self.calculate_container_load(x)}/{obj.capacity})"

            support = self.on.get(x, None)
            container = self.inside.get(x, None)

            if container is not None:
                lines.append(f"{x}{status} is inside {container}")
            elif support is None:
                lines.append(f"{x}{status} is on the table")
            else:
                lines.append(f"{x}{status} is on {support}")

        lines.append(f"Holding: {self.holding}")
        return "\n".join(lines)

    # ----------------------------
    # Actions that can be performed in the world
    # ----------------------------
    def pickup(self, x: str) -> None:
        if self.holding is not None:
            raise RuntimeError("Already holding something.")
        if not self.is_clear(x):
            raise RuntimeError(f"Cannot pick up {x}: not clear.")
        self.holding = x
        self.on[x] = None
        self.inside[x] = None

    def drop(self) -> None:
        """Drop the currently held item onto the table."""
        if self.holding is None:
            raise RuntimeError("Cannot drop anything: not holding an item.")

        item = self.holding
        self.on[item] = None
        self.inside[item] = None
        self.holding = None

    def calculate_container_load(self, container_name: str) -> int:
        return sum(
            self.objects[name].size for name in self.current_contents(container_name)
        )

    def put_on(self, x: str, y: str) -> None:
        if self.holding != x:
            raise RuntimeError(f"Not holding {x}.")
        if not self.is_clear(y):
            raise RuntimeError(f"Cannot place on {y}: {y} not clear.")
        self.on[x] = y
        self.holding = None

    def put_inside(self, x: str, c: str) -> None:
        if self.holding != x:
            raise RuntimeError(f"Not holding {x}.")
        container_obj = self.objects.get(c)
        if not container_obj or not container_obj.is_container:
            raise RuntimeError(f"Cannot put inside {c}: it is not a container.")
        if not self.open_containers.get(c, False):
            raise RuntimeError(f"Cannot put inside {c}: container is closed.")
        if (
            self.calculate_container_load(c) + self.objects[x].size
            >= container_obj.capacity
        ):
            raise RuntimeError(f"Cannot put inside {c}: it does not fit.")

        self.inside[x] = c
        self.on[x] = None
        self.holding = None

    def open_container(self, c: str) -> None:
        if not self.objects[c].is_container:
            raise RuntimeError(f"{c} is not a container.")
        self.open_containers[c] = True

    def close_container(self, c: str) -> None:
        if not self.objects[c].is_container:
            raise RuntimeError(f"{c} is not a container.")
        self.open_containers[c] = False


# ----------------------------------------
# 2) Reference grounding
# ----------------------------------------


def resolve_ref(world: World, color: Optional[str], shape: Optional[str]) -> List[str]:
    matches = []
    for name, obj in world.objects.items():
        if color is not None and obj.color != color:
            continue
        if shape is not None:
            if shape not in ("block", "container") and obj.shape != shape:
                continue
            if shape == "container" and not obj.is_container:
                continue
        matches.append(name)
    return matches


# ----------------------------------------
# 3) Expanded Planner
# ----------------------------------------


def plan_pickup(world: World, x: str) -> List[Tuple[str, str, Optional[str]]]:
    plan = []

    # If it's inside a container, check if container is open
    parent = world.inside.get(x)
    if parent is not None and not world.open_containers.get(parent, False):
        plan.append(("open", parent, None))

    # If something is stacked on it, move it away first
    blocker = world.top_of(x)
    if blocker is not None:
        plan += plan_pickup(world, blocker)
        plan.append(("put_on", blocker, None))  # put blocker on table

    plan.append(("pickup", x, None))
    return plan


def plan_put_on(world: World, x: str, y: str) -> List[Tuple[str, str, Optional[str]]]:
    plan = []
    blocker = world.top_of(y)
    if blocker is not None:
        plan += plan_pickup(world, blocker)
        plan.append(("put_on", blocker, None))
    plan += plan_pickup(world, x)
    plan.append(("put_on", x, y))
    return plan


def plan_put_inside(
    world: World, x: str, c: str
) -> List[Tuple[str, str, Optional[str]]]:
    plan = []
    # If container is closed, add action to open it
    if not world.open_containers.get(c, False):
        plan.append(("open", c, None))

    plan += plan_pickup(world, x)
    plan.append(("put_inside", x, c))
    return plan


def execute_plan(world: World, plan: List[Tuple[str, str, Optional[str]]]) -> None:
    for action, obj, target in plan:
        if action == "pickup":
            world.pickup(obj)
        elif action == "drop":
            world.drop()
        elif action == "put_on":
            if target is None:
                # Putting back on table clears its positions
                if world.holding != obj:
                    raise RuntimeError("Planner/executor mismatch.")
                world.on[obj] = None
                world.inside[obj] = None
                world.holding = None
            else:
                world.put_on(obj, target)
        elif action == "put_inside":
            world.put_inside(obj, target)  # type: ignore
        elif action == "open":
            world.open_container(obj)
        elif action == "close":
            world.close_container(obj)
        else:
            raise ValueError(f"Unknown action: {action}")


# ----------------------------------------
# 4) Expanded Parser
# ----------------------------------------

COLORS = {"red", "green", "blue", "yellow", "brown"}
SHAPES = {"cube", "pyramid", "block", "box", "chest", "barrel"}


def parse_command(text: str) -> dict:

    if text.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        sys.exit(0)

    t = text.lower().replace("?", "").strip()
    tokens = t.split()

    if t.startswith("pick up"):
        color = next((w for w in tokens if w in COLORS), None)
        shape = next((w for w in tokens if w in SHAPES), "block")
        return {"intent": "PICKUP", "ref": {"color": color, "shape": shape}}

    if t.startswith("drop") or t.startswith("let go") or t.startswith("put down"):
        # If they just said "let go" or "put down", we assume the currently held item
        # If they specify an object (e.g., "drop the red cube"), we can parse it:
        color = next((w for w in tokens if w in COLORS), None)
        shape = next((w for w in tokens if w in SHAPES), None)
        return {"intent": "DROP", "ref": {"color": color, "shape": shape}}

    if t.startswith("open") or t.startswith("close"):
        intent = "OPEN" if t.startswith("open") else "CLOSE"
        color = next((w for w in tokens if w in COLORS), None)
        shape = next((w for w in tokens if w in SHAPES), "container")
        return {"intent": intent, "ref": {"color": color, "shape": shape}}

    if t.startswith("put"):
        # Detect whether it is an "on" or "in/inside" placement directive
        if "on" in tokens:
            sep_i = tokens.index("on")
            intent = "PUT_ON"
        elif "in" in tokens:
            sep_i = tokens.index("in")
            intent = "PUT_INSIDE"
        elif "inside" in tokens:
            sep_i = tokens.index("inside")
            intent = "PUT_INSIDE"
        else:
            raise ValueError(
                "Expected structural relation like 'on' or 'in' in command."
            )

        left = tokens[1:sep_i]
        right = tokens[sep_i + 1 :]

        color_x = next((w for w in left if w in COLORS), None)
        shape_x = next((w for w in left if w in SHAPES), "block")
        color_y = next((w for w in right if w in COLORS), None)
        shape_y = next((w for w in right if w in SHAPES), "block")

        return {
            "intent": intent,
            "x": {"color": color_x, "shape": shape_x},
            "y": {"color": color_y, "shape": shape_y},
        }

    raise ValueError("Unknown command format.")


# ----------------------------------------
# 5) Expanded Dialogue Manager
# ----------------------------------------


def choose_unique(matches: List[str], what: str) -> str:
    if not matches:
        raise ValueError(f"I can't find any {what} matching your description.")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguity error: I don't know which {what} you mean: {matches}"
        )
    return matches[0]


def interpret_and_act(world: World, utterance: str) -> None:
    parsed = parse_command(utterance)

    if parsed["intent"] == "PICKUP":
        ref = parsed["ref"]
        matches = resolve_ref(world, ref["color"], ref["shape"])
        x = choose_unique(matches, f"{ref['color'] or ''} {ref['shape']}".strip())
        plan = plan_pickup(world, x)
        print("PLAN:", plan)
        execute_plan(world, plan)
        print(f"OK. Picked up {x}.")

    elif parsed["intent"] == "DROP":
        if world.holding is None:
            raise ValueError("I am not holding anything to let go of.")

        ref = parsed["ref"]
        # If the user specified properties, double-check they match the held item
        if ref["color"] or ref["shape"]:
            matches = resolve_ref(world, ref["color"], ref["shape"])
            if world.holding not in matches:
                raise ValueError(
                    f"I am not holding a {ref['color'] or ''} {ref['shape'] or 'item'}."
                )

        dropped_item = world.holding
        execute_plan(world, [("drop", dropped_item, None)])
        print(f"OK. Let go of {dropped_item} and placed it on the table.")

    elif parsed["intent"] in ("OPEN", "CLOSE"):
        ref = parsed["ref"]
        matches = resolve_ref(world, ref["color"], ref["shape"])
        c = choose_unique(matches, f"{ref['color'] or ''} {ref['shape']}".strip())
        if parsed["intent"] == "OPEN":
            execute_plan(world, [("open", c, None)])
            print(f"OK. Opened {c}.")
        else:
            execute_plan(world, [("close", c, None)])
            print(f"OK. Closed {c}.")

    elif parsed["intent"] == "PUT_ON":
        mx = resolve_ref(world, parsed["x"]["color"], parsed["x"]["shape"])
        my = resolve_ref(world, parsed["y"]["color"], parsed["y"]["shape"])
        x = choose_unique(
            mx, f"{parsed['x']['color'] or ''} {parsed['x']['shape']}".strip()
        )
        y = choose_unique(
            my, f"{parsed['y']['color'] or ''} {parsed['y']['shape']}".strip()
        )
        plan = plan_put_on(world, x, y)
        print("PLAN:", plan)
        execute_plan(world, plan)
        print(f"OK. Put {x} on {y}.")

    elif parsed["intent"] == "PUT_INSIDE":
        mx = resolve_ref(world, parsed["x"]["color"], parsed["x"]["shape"])
        mc = resolve_ref(world, parsed["y"]["color"], parsed["y"]["shape"])
        x = choose_unique(
            mx, f"{parsed['x']['color'] or ''} {parsed['x']['shape']}".strip()
        )
        c = choose_unique(
            mc, f"{parsed['y']['color'] or ''} {parsed['y']['shape']}".strip()
        )
        plan = plan_put_inside(world, x, c)
        print("PLAN:", plan)
        execute_plan(world, plan)
        print(f"OK. Put {x} inside {c}.")


# ----------------------------
# Demo run
# ----------------------------


def demo():
    world = World(
        objects={
            "b1": Obj("b1", "red", "cube"),
            "b2": Obj("b2", "red", "cube"),  # deliberately ambiguous for "red cube"
            "p1": Obj("p1", "blue", "pyramid"),
            "c1": Obj("c1", "green", "cube"),
        },
        on={
            "p1": "b1",  # blue pyramid on red cube (b1)
            "b1": None,
            "b2": None,
            "c1": None,
        },
        inside={},
        open_containers={},
    )

    print("INITIAL WORLD:\n" + world.describe(), "\n")

    # 1) Example that triggers clarification (two red cubes exist)
    try:
        interpret_and_act(world, "put the blue pyramid on the red cube")
    except ValueError as e:
        print("SHRDLU:", e)

    # 2) Disambiguated request
    print(
        "\nDisambiguating by specifying the target name isn't supported by our tiny grammar,"
    )
    print(
        "so we instead remove ambiguity by changing the world (like a controlled micro-domain)."
    )

    # Remove one red cube to resolve ambiguity
    del world.objects["b2"]
    del world.on["b2"]

    print("\nWORLD NOW:\n" + world.describe(), "\n")

    interpret_and_act(world, "put the red cube on the blue pyramid")

    print("\nFINAL WORLD:\n" + world.describe())


def main():

    # ------ World creation cheat sheet

    # COLORS = {"red", "green", "blue", "yellow", "brown"}
    # SHAPES = {"cube", "pyramid", "block", "box", "chest", "barrel"}

    # @dataclass(frozen=True)
    # class Obj:
    #     name: str
    #     color: str
    #     shape: str  # "cube", "pyramid", "box", "chest", "barrel", etc.
    #     is_container: bool = False
    #     capacity: int = 0  # How many items it can hold inside
    #     size: int = 1  # Physical size of the object itself

    # @dataclass
    # class World:
    #     objects: Dict[str, Obj]
    #     on: Dict[str, Optional[str]]  # on[x] = y means x is stacked on y
    #     inside: Dict[str, Optional[str]]  # inside[x] = y means x is inside container y
    #     open_containers: Dict[
    #         str, bool
    #     ]  # maps container name -> True (open) or False (closed)
    #     holding: Optional[str] = None

    # ------ World creation cheat sheet

    world = World(
        objects={
            "b1": Obj("b1", "red", "cube", size=2),
            "p1": Obj("p1", "blue", "pyramid", size=5),
            "c1": Obj("c1", "green", "cube", size=8),
            "c2": Obj("c2", "yellow", "box", size=3, is_container=True, capacity=2),
            "box1": Obj("box1", "brown", "box", is_container=True, capacity=2),
            "chest1": Obj("chest1", "yellow", "chest", is_container=True, capacity=4),
            "barrel1": Obj("barrel1", "red", "barrel", is_container=True, capacity=8),
        },
        on={
            "p1": "b1",  # blue pyramid on red cube (b1)
            "b1": None,
            "c1": None,
        },
        inside={},
        open_containers={},
    )

    while True:
        try:
            interpret_and_act(world, input("Command: "))
            print("\n" + world.describe())
        except (ValueError, RuntimeError) as e:
            print("SHRDLU:", e)


if __name__ == "__main__":
    main()
