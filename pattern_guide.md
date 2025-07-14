# Complete DSA Pattern Recognition Guide: Problem to Algorithm Mapping

## Table of Contents
1. [Arrays & Strings Patterns](#arrays--strings-patterns)
2. [Two Pointers Pattern](#two-pointers-pattern)
3. [Sliding Window Pattern](#sliding-window-pattern)
4. [Fast & Slow Pointers Pattern](#fast--slow-pointers-pattern)
5. [Merge Intervals Pattern](#merge-intervals-pattern)
6. [Cyclic Sort Pattern](#cyclic-sort-pattern)
7. [In-place Reversal Pattern](#in-place-reversal-pattern)
8. [Binary Search Patterns](#binary-search-patterns)
9. [Tree Traversal Patterns](#tree-traversal-patterns)
10. [Graph Traversal Patterns](#graph-traversal-patterns)
11. [Dynamic Programming Patterns](#dynamic-programming-patterns)
12. [Backtracking Patterns](#backtracking-patterns)
13. [Heap/Priority Queue Patterns](#heappriority-queue-patterns)
14. [Stack & Queue Patterns](#stack--queue-patterns)
15. [Greedy Algorithm Patterns](#greedy-algorithm-patterns)
16. [Bit Manipulation Patterns](#bit-manipulation-patterns)
17. [Mathematical Patterns](#mathematical-patterns)
18. [String Manipulation Patterns](#string-manipulation-patterns)

---

## Arrays & Strings Patterns

### Pattern Recognition
**When to use:**
- Problem involves contiguous elements
- Need to track subarrays or substrings
- Finding pairs, triplets, or groups
- Optimization problems on arrays

**Key Indicators:**
- "Find subarray with..."
- "Maximum/minimum sum/product..."
- "Contiguous elements..."
- "Rearrange array..."

### Common Approaches
```cpp
// 1. Prefix Sum - for range sum queries
vector<int> prefixSum(arr.size() + 1, 0);
for (int i = 0; i < arr.size(); i++) {
    prefixSum[i + 1] = prefixSum[i] + arr[i];
}
// Sum from i to j = prefixSum[j+1] - prefixSum[i]

// 2. Kadane's Algorithm - maximum subarray sum
int maxSum = arr[0], currentSum = arr[0];
for (int i = 1; i < arr.size(); i++) {
    currentSum = max(arr[i], currentSum + arr[i]);
    maxSum = max(maxSum, currentSum);
}

// 3. Moore's Voting Algorithm - majority element
int candidate = arr[0], count = 1;
for (int i = 1; i < arr.size(); i++) {
    if (count == 0) {
        candidate = arr[i];
        count = 1;
    } else if (arr[i] == candidate) {
        count++;
    } else {
        count--;
    }
}
```

---

## Two Pointers Pattern

### Pattern Recognition
**When to use:**
- **Sorted array/list** problems
- Finding pairs/triplets with specific sum
- Comparing elements from both ends
- Palindrome checking
- Removing duplicates

**Key Indicators:**
- "Find two/three numbers that..."
- "Remove duplicates from sorted array"
- "Is palindrome?"
- "Container with most water"
- "Valid palindrome after deletion"

### Implementation Templates

```cpp
// 1. Opposite Direction Two Pointers
int left = 0, right = arr.size() - 1;
while (left < right) {
    if (condition_met) {
        // Process or return result
    } else if (need_to_increase_sum) {
        left++;
    } else {
        right--;
    }
}

// 2. Same Direction Two Pointers (Fast & Slow)
int slow = 0;
for (int fast = 0; fast < arr.size(); fast++) {
    if (condition) {
        arr[slow++] = arr[fast];
    }
}

// 3. Three Sum Problem Template
sort(arr.begin(), arr.end());
for (int i = 0; i < arr.size() - 2; i++) {
    if (i > 0 && arr[i] == arr[i-1]) continue; // Skip duplicates
    int left = i + 1, right = arr.size() - 1;
    while (left < right) {
        int sum = arr[i] + arr[left] + arr[right];
        if (sum == target) {
            // Found triplet
            left++; right--;
            // Skip duplicates
            while (left < right && arr[left] == arr[left-1]) left++;
            while (left < right && arr[right] == arr[right+1]) right--;
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
}
```

---

## Sliding Window Pattern

### Pattern Recognition
**When to use:**
- Contiguous subarray/substring problems
- Fixed or variable window size
- Finding optimal window that satisfies condition
- Maximum/minimum in window

**Key Indicators:**
- "Maximum sum subarray of size k"
- "Longest substring with k distinct characters"
- "Minimum window substring"
- "Find all anagrams"
- "Maximum of all subarrays of size k"

### Implementation Templates

```cpp
// 1. Fixed Window Size
int windowSum = 0, maxSum = 0;
int k = 3; // window size
// First window
for (int i = 0; i < k; i++) {
    windowSum += arr[i];
}
maxSum = windowSum;
// Slide window
for (int i = k; i < arr.size(); i++) {
    windowSum = windowSum - arr[i-k] + arr[i];
    maxSum = max(maxSum, windowSum);
}

// 2. Variable Window Size
int left = 0, maxLen = 0;
unordered_map<char, int> charCount;
for (int right = 0; right < s.length(); right++) {
    charCount[s[right]]++;
    
    // Shrink window if condition violated
    while (charCount.size() > k) { // k distinct characters
        charCount[s[left]]--;
        if (charCount[s[left]] == 0) {
            charCount.erase(s[left]);
        }
        left++;
    }
    
    maxLen = max(maxLen, right - left + 1);
}

// 3. Window with Specific Condition
int left = 0, minLen = INT_MAX;
unordered_map<char, int> need, window;
// Initialize need map with target characters
int formed = 0; // Characters with required frequency
for (int right = 0; right < s.length(); right++) {
    char c = s[right];
    window[c]++;
    
    if (need.count(c) && window[c] == need[c]) {
        formed++;
    }
    
    // Try to shrink window
    while (formed == need.size()) {
        if (right - left + 1 < minLen) {
            minLen = right - left + 1;
            // Save window position
        }
        
        char leftChar = s[left];
        window[leftChar]--;
        if (need.count(leftChar) && window[leftChar] < need[leftChar]) {
            formed--;
        }
        left++;
    }
}
```

---

## Fast & Slow Pointers Pattern

### Pattern Recognition
**When to use:**
- Cycle detection in linked list/array
- Finding middle element
- Finding cycle start
- Happy number type problems

**Key Indicators:**
- "Detect if linked list has cycle"
- "Find start of cycle"
- "Middle of linked list"
- "Is happy number?"
- "Find duplicate number"

### Implementation Templates

```cpp
// 1. Cycle Detection (Floyd's Algorithm)
ListNode* hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return slow; // Cycle found
    }
    return nullptr; // No cycle
}

// 2. Find Cycle Start
ListNode* findCycleStart(ListNode* head) {
    ListNode* meeting = hasCycle(head);
    if (!meeting) return nullptr;
    
    ListNode* start = head;
    while (start != meeting) {
        start = start->next;
        meeting = meeting->next;
    }
    return start;
}

// 3. Find Middle Element
ListNode* findMiddle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

// 4. Happy Number
bool isHappy(int n) {
    auto getNext = [](int num) {
        int sum = 0;
        while (num > 0) {
            int digit = num % 10;
            sum += digit * digit;
            num /= 10;
        }
        return sum;
    };
    
    int slow = n, fast = n;
    do {
        slow = getNext(slow);
        fast = getNext(getNext(fast));
    } while (slow != fast);
    
    return slow == 1;
}
```

---

## Merge Intervals Pattern

### Pattern Recognition
**When to use:**
- Dealing with overlapping intervals
- Meeting room problems
- Insert/merge intervals
- Interval scheduling

**Key Indicators:**
- "Merge overlapping intervals"
- "Insert interval"
- "Meeting rooms required"
- "Maximum CPU load"
- "Employee free time"

### Implementation Templates

```cpp
// 1. Merge Intervals
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};
    
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;
    merged.push_back(intervals[0]);
    
    for (int i = 1; i < intervals.size(); i++) {
        if (merged.back()[1] >= intervals[i][0]) {
            // Overlapping, merge
            merged.back()[1] = max(merged.back()[1], intervals[i][1]);
        } else {
            // Non-overlapping, add new
            merged.push_back(intervals[i]);
        }
    }
    return merged;
}

// 2. Insert Interval
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
    vector<vector<int>> result;
    int i = 0;
    
    // Add all intervals before newInterval
    while (i < intervals.size() && intervals[i][1] < newInterval[0]) {
        result.push_back(intervals[i++]);
    }
    
    // Merge overlapping intervals
    while (i < intervals.size() && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = min(newInterval[0], intervals[i][0]);
        newInterval[1] = max(newInterval[1], intervals[i][1]);
        i++;
    }
    result.push_back(newInterval);
    
    // Add remaining intervals
    while (i < intervals.size()) {
        result.push_back(intervals[i++]);
    }
    
    return result;
}

// 3. Meeting Rooms II (Min heap approach)
int minMeetingRooms(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (auto& interval : intervals) {
        if (!minHeap.empty() && minHeap.top() <= interval[0]) {
            minHeap.pop(); // Reuse room
        }
        minHeap.push(interval[1]);
    }
    
    return minHeap.size();
}
```

---

## Cyclic Sort Pattern

### Pattern Recognition
**When to use:**
- Array contains numbers in range [0, n] or [1, n]
- Find missing/duplicate numbers
- Find all missing/duplicate numbers
- Find smallest missing positive

**Key Indicators:**
- "Find missing number"
- "Find duplicate number"
- "Find all duplicates"
- "First missing positive"
- "Find corrupt pair"

### Implementation Templates

```cpp
// 1. Basic Cyclic Sort
void cyclicSort(vector<int>& nums) {
    int i = 0;
    while (i < nums.size()) {
        int correctPos = nums[i] - 1; // For 1-based
        // int correctPos = nums[i]; // For 0-based
        if (nums[i] != nums[correctPos]) {
            swap(nums[i], nums[correctPos]);
        } else {
            i++;
        }
    }
}

// 2. Find Missing Number
int findMissingNumber(vector<int>& nums) {
    int i = 0;
    while (i < nums.size()) {
        if (nums[i] < nums.size() && nums[i] != i) {
            swap(nums[i], nums[nums[i]]);
        } else {
            i++;
        }
    }
    
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != i) return i;
    }
    return nums.size();
}

// 3. Find All Duplicates
vector<int> findDuplicates(vector<int>& nums) {
    vector<int> duplicates;
    int i = 0;
    while (i < nums.size()) {
        int correctPos = nums[i] - 1;
        if (nums[i] != nums[correctPos]) {
            swap(nums[i], nums[correctPos]);
        } else {
            i++;
        }
    }
    
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != i + 1) {
            duplicates.push_back(nums[i]);
        }
    }
    return duplicates;
}
```

---

## In-place Reversal Pattern

### Pattern Recognition
**When to use:**
- Reverse linked list (full or partial)
- Reverse nodes in k-group
- Rotate list
- Swap nodes in pairs

**Key Indicators:**
- "Reverse linked list"
- "Reverse nodes in k-group"
- "Rotate list"
- "Swap every two adjacent nodes"

### Implementation Templates

```cpp
// 1. Reverse Entire Linked List
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// 2. Reverse Sublist (from position m to n)
ListNode* reverseBetween(ListNode* head, int m, int n) {
    if (!head || m == n) return head;
    
    ListNode dummy(0);
    dummy.next = head;
    ListNode* prev = &dummy;
    
    // Move to position m-1
    for (int i = 1; i < m; i++) {
        prev = prev->next;
    }
    
    // Reverse from m to n
    ListNode* curr = prev->next;
    for (int i = m; i < n; i++) {
        ListNode* next = curr->next;
        curr->next = next->next;
        next->next = prev->next;
        prev->next = next;
    }
    
    return dummy.next;
}

// 3. Reverse Nodes in k-Group
ListNode* reverseKGroup(ListNode* head, int k) {
    // Count nodes
    int count = 0;
    ListNode* curr = head;
    while (curr) {
        count++;
        curr = curr->next;
    }
    
    ListNode dummy(0);
    dummy.next = head;
    ListNode* prevGroup = &dummy;
    
    while (count >= k) {
        ListNode* groupStart = prevGroup->next;
        ListNode* groupEnd = groupStart;
        ListNode* nextGroup = groupEnd->next;
        
        // Reverse k nodes
        for (int i = 1; i < k; i++) {
            groupEnd->next = nextGroup->next;
            nextGroup->next = prevGroup->next;
            prevGroup->next = nextGroup;
            nextGroup = groupEnd->next;
        }
        
        prevGroup = groupStart;
        count -= k;
    }
    
    return dummy.next;
}
```

---

## Binary Search Patterns

### Pattern Recognition
**When to use:**
- Sorted array/matrix
- Search in rotated array
- Find peak element
- Search for range
- Minimize/maximize value problems
- Search in infinite array

**Key Indicators:**
- "Find in sorted array"
- "Search in rotated sorted array"
- "Find first/last position"
- "Find peak element"
- "Minimum in rotated sorted array"
- "Search in 2D matrix"
- "Kth smallest element"

### Implementation Templates

```cpp
// 1. Classic Binary Search
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// 2. Find First/Last Occurrence
int findFirst(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1, result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            result = mid;
            right = mid - 1; // Look for earlier occurrence
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return result;
}

// 3. Search in Rotated Sorted Array
int searchRotated(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        
        // Check which half is sorted
        if (arr[left] <= arr[mid]) {
            // Left half is sorted
            if (target >= arr[left] && target < arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            // Right half is sorted
            if (target > arr[mid] && target <= arr[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}

// 4. Find Peak Element
int findPeakElement(vector<int>& arr) {
    int left = 0, right = arr.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] > arr[mid + 1]) {
            right = mid; // Peak is in left half including mid
        } else {
            left = mid + 1; // Peak is in right half
        }
    }
    return left;
}

// 5. Binary Search on Answer Space
int minDays(vector<int>& bloomDay, int m, int k) {
    if ((long)m * k > bloomDay.size()) return -1;
    
    int left = *min_element(bloomDay.begin(), bloomDay.end());
    int right = *max_element(bloomDay.begin(), bloomDay.end());
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        // Check if we can make m bouquets in mid days
        int bouquets = 0, flowers = 0;
        for (int day : bloomDay) {
            if (day <= mid) {
                flowers++;
                if (flowers == k) {
                    bouquets++;
                    flowers = 0;
                }
            } else {
                flowers = 0;
            }
        }
        
        if (bouquets >= m) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}
```

---

## Tree Traversal Patterns

### Pattern Recognition
**When to use:**
- Need specific order of node visitation
- Level-by-level processing
- Path problems
- Tree construction/serialization

**Key Indicators:**
- "Traverse in order/preorder/postorder"
- "Level order traversal"
- "Zigzag traversal"
- "Path sum"
- "Lowest common ancestor"
- "Serialize/deserialize tree"

### Implementation Templates

```cpp
// 1. DFS Traversals
// Inorder (Left -> Root -> Right)
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Preorder (Root -> Left -> Right)
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// Postorder (Left -> Right -> Root)
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);
    postorder(root->right, result);
    result.push_back(root->val);
}

// 2. BFS Level Order Traversal
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> level;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}

// 3. Path Sum Pattern
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;
    if (!root->left && !root->right) {
        return root->val == targetSum;
    }
    return hasPathSum(root->left, targetSum - root->val) || 
           hasPathSum(root->right, targetSum - root->val);
}

// 4. All Paths Pattern
void findPaths(TreeNode* root, int targetSum, vector<int>& path, 
                vector<vector<int>>& paths) {
    if (!root) return;
    
    path.push_back(root->val);
    
    if (!root->left && !root->right && root->val == targetSum) {
        paths.push_back(path);
    } else {
        findPaths(root->left, targetSum - root->val, path, paths);
        findPaths(root->right, targetSum - root->val, path, paths);
    }
    
    path.pop_back(); // Backtrack
}

// 5. Tree Diameter Pattern
int diameterOfBinaryTree(TreeNode* root) {
    int diameter = 0;
    
    function<int(TreeNode*)> height = [&](TreeNode* node) -> int {
        if (!node) return 0;
        int leftHeight = height(node->left);
        int rightHeight = height(node->right);
        diameter = max(diameter, leftHeight + rightHeight);
        return 1 + max(leftHeight, rightHeight);
    };
    
    height(root);
    return diameter;
}
```

---

## Graph Traversal Patterns

### Pattern Recognition
**When to use:**
- Find path between nodes
- Detect cycles
- Find connected components
- Topological sorting
- Shortest path problems
- Bipartite checking

**Key Indicators:**
- "Find if path exists"
- "Number of islands/connected components"
- "Is graph bipartite?"
- "Course schedule (dependencies)"
- "Shortest path"
- "Minimum spanning tree"

### Implementation Templates

```cpp
// 1. DFS Template
void dfs(vector<vector<int>>& graph, int node, vector<bool>& visited) {
    visited[node] = true;
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfs(graph, neighbor, visited);
        }
    }
}

// 2. BFS Template
void bfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    queue<int> q;
    
    q.push(start);
    visited[start] = true;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

// 3. Cycle Detection (Directed Graph)
bool hasCycle(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> state(n, 0); // 0: unvisited, 1: visiting, 2: visited
    
    function<bool(int)> dfs = [&](int node) -> bool {
        state[node] = 1;
        
        for (int neighbor : graph[node]) {
            if (state[neighbor] == 1) return true; // Back edge
            if (state[neighbor] == 0 && dfs(neighbor)) return true;
        }
        
        state[node] = 2;
        return false;
    };
    
    for (int i = 0; i < n; i++) {
        if (state[i] == 0 && dfs(i)) return true;
    }
    return false;
}

// 4. Topological Sort (Kahn's Algorithm - BFS)
vector<int> topologicalSort(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> indegree(n, 0);
    
    // Calculate indegrees
    for (int i = 0; i < n; i++) {
        for (int neighbor : graph[i]) {
            indegree[neighbor]++;
        }
    }
    
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) q.push(i);
    }
    
    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        
        for (int neighbor : graph[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    return result.size() == n ? result : vector<int>();
}

// 5. Bipartite Check
bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> color(n, -1);
    
    function<bool(int, int)> dfs = [&](int node, int c) -> bool {
        color[node] = c;
        
        for (int neighbor : graph[node]) {
            if (color[neighbor] == -1) {
                if (!dfs(neighbor, 1 - c)) return false;
            } else if (color[neighbor] == color[node]) {
                return false;
            }
        }
        return true;
    };
    
    for (int i = 0; i < n; i++) {
        if (color[i] == -1) {
            if (!dfs(i, 0)) return false;
        }
    }
    return true;
}

// 6. Dijkstra's Shortest Path
vector<int> dijkstra(vector<vector<pair<int,int>>>& graph, int start) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    
    dist[start] = 0;
    pq.push({0, start});
    
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        
        if (d > dist[u]) continue;
        
        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}
```

---

## Dynamic Programming Patterns

### Pattern Recognition
**When to use:**
- Optimal substructure property
- Overlapping subproblems
- Decisions at each step
- Count ways/paths
- Min/max optimization

**Key Indicators:**
- "Number of ways to..."
- "Minimum/maximum cost to reach..."
- "Longest/shortest..."
- "Count distinct ways..."
- "Is it possible to reach..."
- "Optimal strategy for..."

### Major DP Patterns

#### 1. Linear DP (1D)
```cpp
// Fibonacci Pattern
int fib(int n) {
    if (n <= 1) return n;
    vector<int> dp(n + 1);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// House Robber Pattern
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    
    int prev2 = nums[0];
    int prev1 = max(nums[0], nums[1]);
    
    for (int i = 2; i < n; i++) {
        int curr = max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// Climbing Stairs with Cost
int minCostClimbingStairs(vector<int>& cost) {
    int n = cost.size();
    for (int i = 2; i < n; i++) {
        cost[i] += min(cost[i-1], cost[i-2]);
    }
    return min(cost[n-1], cost[n-2]);
}
```

#### 2. Grid DP (2D)
```cpp
// Unique Paths
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}

// Minimum Path Sum
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    
    for (int i = 1; i < m; i++) {
        grid[i][0] += grid[i-1][0];
    }
    for (int j = 1; j < n; j++) {
        grid[0][j] += grid[0][j-1];
    }
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            grid[i][j] += min(grid[i-1][j], grid[i][j-1]);
        }
    }
    return grid[m-1][n-1];
}
```

#### 3. Knapsack Patterns
```cpp
// 0/1 Knapsack
int knapsack(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i-1][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][W];
}

// Unbounded Knapsack (Coin Change)
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}

// Subset Sum
bool canPartition(vector<int>& nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2 != 0) return false;
    
    int target = sum / 2;
    vector<bool> dp(target + 1, false);
    dp[0] = true;
    
    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    return dp[target];
}
```

#### 4. String DP Patterns
```cpp
// Longest Common Subsequence
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.size(), n = text2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}

// Edit Distance
int minDistance(string word1, string word2) {
    int m = word1.size(), n = word2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j],    // delete
                                   dp[i][j-1],      // insert
                                   dp[i-1][j-1]});  // replace
            }
        }
    }
    return dp[m][n];
}

// Longest Palindromic Subsequence
int longestPalindromeSubseq(string s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    
    for (int i = 0; i < n; i++) {
        dp[i][i] = 1;
    }
    
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                dp[i][j] = dp[i+1][j-1] + 2;
            } else {
                dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
            }
        }
    }
    return dp[0][n-1];
}
```

#### 5. State Machine DP
```cpp
// Best Time to Buy and Sell Stock with Cooldown
int maxProfit(vector<int>& prices) {
    if (prices.empty()) return 0;
    
    int hold = -prices[0];
    int sold = 0;
    int rest = 0;
    
    for (int i = 1; i < prices.size(); i++) {
        int prevHold = hold;
        int prevSold = sold;
        int prevRest = rest;
        
        hold = max(prevHold, prevRest - prices[i]);
        sold = prevHold + prices[i];
        rest = max(prevRest, prevSold);
    }
    
    return max(sold, rest);
}

// Stock with Transaction Fee
int maxProfit(vector<int>& prices, int fee) {
    int cash = 0;
    int hold = -prices[0];
    
    for (int i = 1; i < prices.size(); i++) {
        int prevCash = cash;
        cash = max(cash, hold + prices[i] - fee);
        hold = max(hold, prevCash - prices[i]);
    }
    
    return cash;
}
```

#### 6. Interval DP
```cpp
// Burst Balloons
int maxCoins(vector<int>& nums) {
    int n = nums.size();
    nums.insert(nums.begin(), 1);
    nums.push_back(1);
    
    vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
    
    for (int len = 1; len <= n; len++) {
        for (int left = 1; left <= n - len + 1; left++) {
            int right = left + len - 1;
            for (int k = left; k <= right; k++) {
                dp[left][right] = max(dp[left][right],
                    dp[left][k-1] + nums[left-1] * nums[k] * nums[right+1] + dp[k+1][right]);
            }
        }
    }
    
    return dp[1][n];
}
```

---

## Backtracking Patterns

### Pattern Recognition
**When to use:**
- Generate all possible solutions
- Find valid combinations/permutations
- Constraint satisfaction problems
- Decision tree exploration
- Puzzle solving

**Key Indicators:**
- "Find all possible..."
- "Generate all combinations/permutations"
- "N-Queens problem"
- "Sudoku solver"
- "Word search"
- "Generate parentheses"
- "Subset generation"

### Implementation Templates

```cpp
// 1. Generate All Subsets
void generateSubsets(vector<int>& nums, int index, vector<int>& current,
                     vector<vector<int>>& result) {
    result.push_back(current);
    
    for (int i = index; i < nums.size(); i++) {
        current.push_back(nums[i]);
        generateSubsets(nums, i + 1, current, result);
        current.pop_back(); // Backtrack
    }
}

// 2. Generate Permutations
void permute(vector<int>& nums, vector<bool>& used, vector<int>& current,
             vector<vector<int>>& result) {
    if (current.size() == nums.size()) {
        result.push_back(current);
        return;
    }
    
    for (int i = 0; i < nums.size(); i++) {
        if (used[i]) continue;
        
        current.push_back(nums[i]);
        used[i] = true;
        permute(nums, used, current, result);
        current.pop_back();
        used[i] = false;
    }
}

// 3. Combination Sum
void combinationSum(vector<int>& candidates, int target, int start,
                    vector<int>& current, vector<vector<int>>& result) {
    if (target == 0) {
        result.push_back(current);
        return;
    }
    
    for (int i = start; i < candidates.size(); i++) {
        if (candidates[i] > target) break; // Pruning
        
        current.push_back(candidates[i]);
        combinationSum(candidates, target - candidates[i], i, current, result);
        current.pop_back();
    }
}

// 4. N-Queens
bool isSafe(vector<string>& board, int row, int col) {
    // Check column
    for (int i = 0; i < row; i++) {
        if (board[i][col] == 'Q') return false;
    }
    
    // Check diagonal (top-left to bottom-right)
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q') return false;
    }
    
    // Check anti-diagonal (top-right to bottom-left)
    for (int i = row - 1, j = col + 1; i >= 0 && j < board.size(); i--, j++) {
        if (board[i][j] == 'Q') return false;
    }
    
    return true;
}

void solveNQueens(vector<string>& board, int row, vector<vector<string>>& solutions) {
    if (row == board.size()) {
        solutions.push_back(board);
        return;
    }
    
    for (int col = 0; col < board.size(); col++) {
        if (isSafe(board, row, col)) {
            board[row][col] = 'Q';
            solveNQueens(board, row + 1, solutions);
            board[row][col] = '.'; // Backtrack
        }
    }
}

// 5. Word Search
bool exist(vector<vector<char>>& board, string& word, int i, int j, int index) {
    if (index == word.length()) return true;
    if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() ||
        board[i][j] != word[index]) {
        return false;
    }
    
    char temp = board[i][j];
    board[i][j] = '#'; // Mark visited
    
    bool found = exist(board, word, i+1, j, index+1) ||
                 exist(board, word, i-1, j, index+1) ||
                 exist(board, word, i, j+1, index+1) ||
                 exist(board, word, i, j-1, index+1);
    
    board[i][j] = temp; // Backtrack
    return found;
}

// 6. Generate Parentheses
void generateParenthesis(int open, int close, string& current,
                        vector<string>& result) {
    if (open == 0 && close == 0) {
        result.push_back(current);
        return;
    }
    
    if (open > 0) {
        current.push_back('(');
        generateParenthesis(open - 1, close, current, result);
        current.pop_back();
    }
    
    if (close > open) {
        current.push_back(')');
        generateParenthesis(open, close - 1, current, result);
        current.pop_back();
    }
}
```

---

## Heap/Priority Queue Patterns

### Pattern Recognition
**When to use:**
- Find K largest/smallest elements
- Merge K sorted arrays
- Median of stream
- Task scheduling
- Dijkstra's algorithm

**Key Indicators:**
- "K largest/smallest"
- "Merge K sorted..."
- "Median from data stream"
- "Schedule tasks"
- "Kth frequent element"
- "Closest points to origin"

### Implementation Templates

```cpp
// 1. K Largest Elements
vector<int> findKLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }
    
    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top());
        minHeap.pop();
    }
    return result;
}

// 2. Merge K Sorted Lists
ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto comp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(comp)> minHeap(comp);
    
    for (ListNode* list : lists) {
        if (list) minHeap.push(list);
    }
    
    ListNode dummy(0);
    ListNode* tail = &dummy;
    
    while (!minHeap.empty()) {
        ListNode* node = minHeap.top();
        minHeap.pop();
        tail->next = node;
        tail = tail->next;
        
        if (node->next) {
            minHeap.push(node->next);
        }
    }
    
    return dummy.next;
}

// 3. Find Median from Data Stream
class MedianFinder {
    priority_queue<int> maxHeap; // Left half
    priority_queue<int, vector<int>, greater<int>> minHeap; // Right half
    
public:
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();
        
        if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};

// 4. Task Scheduler
int leastInterval(vector<char>& tasks, int n) {
    unordered_map<char, int> freq;
    for (char task : tasks) {
        freq[task]++;
    }
    
    priority_queue<int> maxHeap;
    for (auto& p : freq) {
        maxHeap.push(p.second);
    }
    
    int cycles = 0;
    while (!maxHeap.empty()) {
        vector<int> temp;
        for (int i = 0; i <= n; i++) {
            if (!maxHeap.empty()) {
                temp.push_back(maxHeap.top());
                maxHeap.pop();
            }
        }
        
        for (int count : temp) {
            if (--count > 0) {
                maxHeap.push(count);
            }
        }
        
        cycles += maxHeap.empty() ? temp.size() : n + 1;
    }
    
    return cycles;
}
```

---

## Stack & Queue Patterns

### Pattern Recognition
**When to use:**
- Matching parentheses/brackets
- Next greater/smaller element
- Expression evaluation
- Histogram problems
- Simplify path
- BFS traversal

**Key Indicators:**
- "Valid parentheses"
- "Next greater element"
- "Largest rectangle"
- "Evaluate expression"
- "Simplify path"
- "Min stack"

### Implementation Templates

```cpp
// 1. Valid Parentheses
bool isValid(string s) {
    stack<char> st;
    unordered_map<char, char> pairs = {
        {')', '('}, {']', '['}, {'}', '{'}
    };
    
    for (char c : s) {
        if (pairs.count(c)) {
            if (st.empty() || st.top() != pairs[c]) {
                return false;
            }
            st.pop();
        } else {
            st.push(c);
        }
    }
    return st.empty();
}

// 2. Next Greater Element
vector<int> nextGreaterElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st; // Store indices
    
    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] < nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    return result;
}

// 3. Largest Rectangle in Histogram
int largestRectangleArea(vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    heights.push_back(0); // Sentinel
    
    for (int i = 0; i < heights.size(); i++) {
        while (!st.empty() && heights[st.top()] > heights[i]) {
            int h = heights[st.top()];
            st.pop();
            int w = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, h * w);
        }
        st.push(i);
    }
    return maxArea;
}

// 4. Min Stack
class MinStack {
    stack<pair<int, int>> st; // {value, min_so_far}
    
public:
    void push(int val) {
        int minVal = st.empty() ? val : min(val, st.top().second);
        st.push({val, minVal});
    }
    
    void pop() {
        st.pop();
    }
    
    int top() {
        return st.top().first;
    }
    
    int getMin() {
        return st.top().second;
    }
};

// 5. Monotonic Stack Pattern
vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st; // Decreasing stack
    
    for (int i = 0; i < n; i++) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            int idx = st.top();
            st.pop();
            result[idx] = i - idx;
        }
        st.push(i);
    }
    return result;
}
```

---

## Greedy Algorithm Patterns

### Pattern Recognition
**When to use:**
- Local optimal leads to global optimal
- Activity selection
- Interval scheduling
- Huffman coding
- Minimum spanning tree

**Key Indicators:**
- "Maximum number of..."
- "Minimum cost to..."
- "Activity selection"
- "Job scheduling"
- "Minimum number of coins"
- "Gas station"

### Implementation Templates

```cpp
// 1. Activity Selection
int maxActivities(vector<pair<int, int>>& activities) {
    sort(activities.begin(), activities.end(), 
         [](auto& a, auto& b) { return a.second < b.second; });
    
    int count = 1;
    int lastEnd = activities[0].second;
    
    for (int i = 1; i < activities.size(); i++) {
        if (activities[i].first >= lastEnd) {
            count++;
            lastEnd = activities[i].second;
        }
    }
    return count;
}

// 2. Jump Game
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    
    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
        if (maxReach >= nums.size() - 1) return true;
    }
    return true;
}

// 3. Gas Station
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int totalGas = 0, totalCost = 0;
    int start = 0, tank = 0;
    
    for (int i = 0; i < gas.size(); i++) {
        totalGas += gas[i];
        totalCost += cost[i];
        tank += gas[i] - cost[i];
        
        if (tank < 0) {
            start = i + 1;
            tank = 0;
        }
    }
    
    return totalGas >= totalCost ? start : -1;
}

// 4. Non-overlapping Intervals
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;
    
    sort(intervals.begin(), intervals.end(),
         [](auto& a, auto& b) { return a[1] < b[1]; });
    
    int count = 0;
    int lastEnd = intervals[0][1];
    
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] < lastEnd) {
            count++; // Remove this interval
        } else {
            lastEnd = intervals[i][1];
        }
    }
    return count;
}
```

---

## Bit Manipulation Patterns

### Pattern Recognition
**When to use:**
- Count bits
- XOR properties
- Power of 2
- Find missing/duplicate
- Subset generation using bits

**Key Indicators:**
- "Single number"
- "Count bits"
- "Power of two"
- "Bitwise AND/OR/XOR"
- "Find missing number"

### Implementation Templates

```cpp
// 1. Count Set Bits
int countBits(int n) {
    int count = 0;
    while (n) {
        count++;
        n &= (n - 1); // Remove rightmost set bit
    }
    return count;
}

// 2. Single Number (XOR)
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}

// 3. Power of Two
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// 4. Generate Subsets using Bits
vector<vector<int>> subsets(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> result;
    
    for (int mask = 0; mask < (1 << n); mask++) {
        vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                subset.push_back(nums[i]);
            }
        }
        result.push_back(subset);
    }
    return result;
}

// 5. Find Missing Number
int missingNumber(vector<int>& nums) {
    int n = nums.size();
    int xorSum = n;
    
    for (int i = 0; i < n; i++) {
        xorSum ^= i ^ nums[i];
    }
    return xorSum;
}
```

---

## Mathematical Patterns

### Pattern Recognition
**When to use:**
- GCD/LCM problems
- Prime numbers
- Factorial/combinations
- Mathematical sequences
- Modular arithmetic

**Key Indicators:**
- "Greatest common divisor"
- "Least common multiple"
- "Is prime?"
- "Count primes"
- "Factorial"
- "nCr combinations"

### Implementation Templates

```cpp
// 1. GCD (Euclidean Algorithm)
int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// 2. LCM
int lcm(int a, int b) {
    return (a * b) / gcd(a, b);
}

// 3. Sieve of Eratosthenes
vector<bool> sieve(int n) {
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;
    
    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }
    return isPrime;
}

// 4. Fast Power
long long fastPower(long long base, long long exp, long long mod) {
    long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp /= 2;
    }
    return result;
}

// 5. nCr Calculation
int nCr(int n, int r) {
    if (r > n - r) r = n - r; // Optimization
    
    long long result = 1;
    for (int i = 0; i < r; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}
```

---

## String Manipulation Patterns

### Pattern Recognition
**When to use:**
- Pattern matching
- String transformation
- Anagram problems
- Palindrome checking
- String parsing

**Key Indicators:**
- "Is anagram?"
- "Group anagrams"
- "Longest palindrome"
- "String matching"
- "Word pattern"
- "Decode string"

### Implementation Templates

```cpp
// 1. KMP Pattern Matching
vector<int> computeLPS(string pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int len = 0, i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else if (len != 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

int KMPSearch(string text, string pattern) {
    vector<int> lps = computeLPS(pattern);
    int n = text.length(), m = pattern.length();
    int i = 0, j = 0;
    
    while (i < n) {
        if (text[i] == pattern[j]) {
            i++; j++;
        }
        
        if (j == m) {
            return i - j; // Pattern found
        } else if (i < n && text[i] != pattern[j]) {
            if (j != 0) j = lps[j - 1];
            else i++;
        }
    }
    return -1;
}

// 2. Check Anagram
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;
    
    unordered_map<char, int> count;
    for (int i = 0; i < s.length(); i++) {
        count[s[i]]++;
        count[t[i]]--;
    }
    
    for (auto& p : count) {
        if (p.second != 0) return false;
    }
    return true;
}

// 3. Group Anagrams
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;
    
    for (string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    
    vector<vector<string>> result;
    for (auto& p : groups) {
        result.push_back(p.second);
    }
    return result;
}

// 4. Longest Palindromic Substring (Expand Around Center)
string longestPalindrome(string s) {
    if (s.empty()) return "";
    
    auto expandAroundCenter = [&](int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    };
    
    int start = 0, maxLen = 0;
    for (int i = 0; i < s.length(); i++) {
        int len1 = expandAroundCenter(i, i);
        int len2 = expandAroundCenter(i, i + 1);
        int len = max(len1, len2);
        
        if (len > maxLen) {
            maxLen = len;
            start = i - (len - 1) / 2;
        }
    }
    
    return s.substr(start, maxLen);
}

// 5. Word Pattern
bool wordPattern(string pattern, string s) {
    unordered_map<char, string> charToWord;
    unordered_map<string, char> wordToChar;
    
    istringstream iss(s);
    string word;
    int i = 0;
    
    while (iss >> word) {
        if (i >= pattern.length()) return false;
        char c = pattern[i];
        
        if (charToWord.count(c) && charToWord[c] != word) return false;
        if (wordToChar.count(word) && wordToChar[word] != c) return false;
        
        charToWord[c] = word;
        wordToChar[word] = c;
        i++;
    }
    
    return i == pattern.length();
}
```

---

## Summary: Quick Pattern Recognition Guide

### Array Problems
- **Sorted array** → Two Pointers / Binary Search
- **Subarray/Substring** → Sliding Window
- **In-place modification** → Two Pointers (same direction)
- **K elements** → Heap
- **Range queries** → Prefix Sum

### Linked List Problems
- **Cycle detection** → Fast & Slow Pointers
- **Reverse** → In-place Reversal
- **Merge** → Two Pointers
- **Find middle** → Fast & Slow Pointers

### Tree Problems
- **Level-wise** → BFS
- **Path problems** → DFS
- **Serialize/Deserialize** → BFS/DFS
- **Ancestor problems** → DFS with parent tracking

### Graph Problems
- **Shortest path (unweighted)** → BFS
- **All paths** → DFS/Backtracking
- **Cycle detection** → DFS with states
- **Connected components** → DFS/BFS/Union-Find
- **Topological order** → DFS/Kahn's Algorithm

### Dynamic Programming Indicators
- **Optimal substructure** → DP
- **Overlapping subproblems** → DP with memoization
- **Count ways** → DP
- **Min/Max path** → DP
- **Decision making** → DP

### Backtracking Indicators
- **Generate all** → Backtracking
- **Find valid combinations** → Backtracking
- **Constraint satisfaction** → Backtracking
- **Puzzle solving** → Backtracking

### Greedy Indicators
- **Local optimal = Global optimal** → Greedy
- **Activity selection** → Greedy
- **Interval problems** → Sort + Greedy

This comprehensive guide covers the major DSA patterns you'll encounter in coding interviews. Practice recognizing these patterns in problems, and you'll be able to quickly identify the appropriate algorithm to use.