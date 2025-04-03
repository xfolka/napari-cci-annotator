from threading import Lock

class EquivalenceList:
    
    def __init__(self):
        self._the_list = list()
        self._mutex = Lock()
        self.group_id_map = {}


    def add_eqvivalence_pair(self, id1, id2):
        with self._mutex:
            self._the_list.append((id1,id2))
            
        
    def get_equivalent_id(self, id):
        with self._mutex:
            return self.group_id_map.get(id,id) #if the key does not exist just return the key iteself

    def group_ids(self):
        
        with self._mutex:
            parent = {}
            rank = {}

            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 1
                if parent[x] != x:
                    parent[x] = find(parent[x])  # Path compression
                return parent[x]

            def union(x, y):
                x_root = find(x)
                y_root = find(y)
                if x_root == y_root:
                    return
                # Union by rank to keep trees shallow
                if rank[x_root] < rank[y_root]:
                    parent[x_root] = y_root
                else:
                    parent[y_root] = x_root
                    if rank[x_root] == rank[y_root]:
                        rank[x_root] += 1

            # Process all equivalence pairs
            #for a, b in equivalence_pairs:
            for a, b in self._the_list:
                union(a, b)

            # Find all unique IDs
            all_ids = set()
            #for a, b in equivalence_pairs:
            for a, b in self._the_list:
                all_ids.add(a)
                all_ids.add(b)
            all_ids = list(all_ids)

            # Group by root and find minimum ID per group
            groups = {}
            for id in all_ids:
                root = find(id)
                if root not in groups:
                    groups[root] = []
                groups[root].append(id)

            # Assign group ID as the minimum ID in each group
            
            for root, members in groups.items():
                group_id = min(members)
                for member in members:
                    self.group_id_map[member] = group_id

            #return self.group_id_map

# Example usage:
# equivalence_pairs = [(1, 2), (3, 7), (7, 5), (2, 8)]
# result = group_ids(equivalence_pairs)
#print(result)  # Output: {1: 1, 2: 1, 3: 3, 5: 3, 7: 3, 8: 1}

# el = EquivalenceList()
# el.add_eqvivalence_pair(1,2)
# el.add_eqvivalence_pair(3,7)
# el.add_eqvivalence_pair(7,5)
# el.add_eqvivalence_pair(2,8)

# el.group_ids()

# for i in range(1,9):
#     print(f"id: {i}, eq: {el.get_equivalent_id(i)}")
    


