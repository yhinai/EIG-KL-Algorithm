#!/usr/bin/env python3
import random
import argparse
from typing import List, Dict
import array

class FastCircuitGenerator:
    def __init__(self, size_multiplier: float = 1.0):
        """Initialize with simplified size distribution for speed."""
        self.size_multiplier = size_multiplier
        # Simplified distribution focusing on most common sizes
        self.size_probabilities = [
            (2, 84),  # Most common
            (3, 2),
            (4, 6),
            (5, 2),
            (6, 4),
            (8, 2),  # Combine remaining probabilities
        ]
        self.total_prob = sum(prob for _, prob in self.size_probabilities)

    def _choose_net_size(self) -> int:
        """Fast net size selection using simplified distribution."""
        r = random.uniform(0, self.total_prob)
        current = 0
        for size, prob in self.size_probabilities:
            current += prob
            if r <= current:
                return size
        return 2

    def _select_nodes_for_net(self, available_nodes: array.array, size: int) -> List[int]:
        """Simplified node selection without weights."""
        if size > len(available_nodes):
            size = len(available_nodes)
        
        # Fast random selection without weights
        selected_indices = random.sample(range(len(available_nodes)), size)
        return sorted(available_nodes[i] for i in selected_indices)

    def generate_circuit(self) -> tuple[List[List[int]], int, int]:
        """Generate circuit with simplified node selection."""
        num_nodes = int(201920 * self.size_multiplier)
        num_nets = int(210613 * self.size_multiplier)
        
        # Use array for faster operations
        available_nodes = array.array('i', range(1, num_nodes + 1))
        
        # Pre-allocate nets list for better performance
        nets = []
        nets_append = nets.append  # Local reference for faster append
        
        for _ in range(num_nets):
            net_size = self._choose_net_size()
            net_nodes = self._select_nodes_for_net(available_nodes, net_size)
            if net_nodes:
                nets_append(net_nodes)
        
        return nets, len(nets), num_nodes

    def write_to_file(self, filename: str):
        """Write generated circuit to file with minimal overhead."""
        nets, num_nets, num_nodes = self.generate_circuit()
        
        with open(filename, 'w') as f:
            f.write(f"{num_nets} {num_nodes}\n")
            for net in nets:
                f.write(" ".join(map(str, net)) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate a circuit in .hgr format')
    parser.add_argument('size', type=float, help='Size multiplier (1.0 = reference size)')
    parser.add_argument('--output', '-o', type=str, default='generated_circuit.hgr',
                       help='Output filename (default: generated_circuit.hgr)')
    
    args = parser.parse_args()
    
    generator = FastCircuitGenerator(args.size)
    generator.write_to_file(args.output)
    
    print(f"Generated circuit written to: {args.output}")
    print(f"Circuit size: {args.size}x reference")


if __name__ == "__main__":
    main()