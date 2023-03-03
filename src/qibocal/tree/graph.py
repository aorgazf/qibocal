from typing import List

import networkx as nx

from qibocal.tree.runcard import Action


class Graph(nx.DiGraph):
    @classmethod
    def load(cls, actions: List[dict]):
        return cls.from_actions([Action(**d) for d in actions])

    @classmethod
    def from_actions(cls, actions: List[Action]):
        dig = cls()

        for action in actions:
            dig.add_node(action.id, action=action)

        for node, data in dig.nodes.items():
            action: Action = data["action"]

            if action.main is not None:
                dig.add_edge(node, action.main, main=True)

            assert action.next is not None
            for succ in action.next:
                dig.add_edge(node, succ)

        return dig
