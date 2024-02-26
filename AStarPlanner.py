import numpy as np

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.cost = 0

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)
    
    def __lt__(self, other):
        return self.cost < other.cost

class AStarPlanner(object):    
    def __init__(self, planning_env):
        self.planning_env = planning_env

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = [] 

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []
        
        step = 5
        actions = [[step, step], [step, 0], [step, -step], [0, -step], [-step, -step], [-step, 0], [-step, step], [0, step]]
        weight = 1

        open_set = {}
        closed_set = set()

        start_node = Node(None, self.planning_env.start)
        start_node.cost = 0

        start_key = tuple(start_node.position)
        open_set.setdefault(start_key, {"node": start_node, "cost": 0})

        while open_set:
            current_key = min(open_set, key=lambda k: open_set[k]["cost"])
            current_entry = open_set.pop(current_key)
            current_node = current_entry["node"]
            closed_set.add(tuple(current_node.position))
            self.expanded_nodes.append(current_node.position)

            if np.array_equal(current_node.position, self.planning_env.goal):
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return np.array(path[::-1])

            for action in actions:
                child_position = np.array([current_node.position[0] + action[0], current_node.position[1] + action[1]])
                child_key = tuple(child_position)

                if (
                    not self.planning_env.state_validity_checker(child_position)
                    or not self.planning_env.edge_validity_checker(child_position, current_node.position)
                    or child_key in closed_set
                ):
                    continue

                child = Node(current_node, child_position)
                cost = current_node.cost + self.planning_env.compute_distance(current_node.position, child.position)
                total_cost = cost + weight * self.planning_env.compute_heuristic(child.position)

                if child_key not in open_set or total_cost < open_set[child_key]["cost"]:
                    child.cost = cost
                    open_set[child_key] = {"node": child, "cost": total_cost}

        return np.array(plan)


    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
