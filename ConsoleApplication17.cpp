// ConsoleApplication17.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<string>
#include<math.h>
#include<stack>
#include<string>
#include<queue>
#include<vector>
#include<algorithm>
using namespace std;

//矩阵中的路径;
//写代码不要写中文;又忘了
bool hasPathCore(char *matrix,int rows,int cols,int row,int col,char *str,
	int &pathLength,bool *visited)
{
	if (str[pathLength] == '\0')
		return true;
	bool hasPathFlag = false;
	if (row < rows&&row >= 0 && col < cols&&col >= 0 && matrix[row*cols + col] == str[pathLength] && visited[row*cols + col] == false)
	{
		//make_move
		++pathLength;
		visited[row*cols + col] = true;
		//backtrack；
		hasPathFlag = hasPathCore(matrix, rows, cols, row + 1, col, str, pathLength, visited) 
			|| hasPathCore(matrix, rows, cols, row - 1, col, str, pathLength, visited)
			|| hasPathCore(matrix, rows, cols, row, col+1, str, pathLength, visited)
			|| hasPathCore(matrix, rows, cols, row, col-1, str, pathLength, visited);
		//unmake_move;
		if (!hasPathFlag)
		{
			pathLength--;
			visited[row*cols + col] = false;
		}
	}
	return hasPathFlag;//否则，无解
}
bool hasPath(char *matrix, int rows, int cols, char *str)
{
	//类型检查
	if (matrix == NULL&&rows < 0 && cols < 0 && str == NULL)
		return true;
	bool *visited = new bool[rows*cols];
	memset(visited,0,rows*cols);//memset记住慎用 blog上有 但没测试过
	int pathLength = 0;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			if (hasPathCore(matrix, rows, cols, row, col, str, pathLength, visited))
				return true;
		}
	}
	return false;
}

//机器人的路径
//btw:其实和上面的大同小异，就是算法复杂度有所提升
int getDigitSum(int number)
{
	int sum = 0;
	while (number > 0)
	{
		sum += number % 10;
		number /= 10;
	}
	return sum;
}
bool check(int threshold, int row, int col, int rows, int cols, bool *visited)
{
	if (row < rows&&row >= 0 && col < cols&&col >= 0 && visited[row*cols + col] == false && getDigitSum(row) + getDigitSum(col) < threshold)
		return true;
	return false;
}
int movingcountCore(int threshold,int rows,int cols,int row,int col,bool *visited)
{
	int count = 0;
	if (check(threshold, row, col, rows, cols, visited))
	{
		visited[row*cols + col] = true;
		count = 1 + movingcountCore(threshold, rows, cols, row + 1, col, visited) + movingcountCore(threshold, rows, cols, row - 1, col, visited)
			+ movingcountCore(threshold, rows, cols, row, col + 1, visited) + movingcountCore(threshold, rows, cols, row, col - 1, visited);
	}
	return count;
}
int movingcount(int threshold,int rows,int cols)
{
	bool *visited = new bool[rows*cols];
	//这一块不是清零 那么就不要用memset
	for (int i = 0; i < rows*cols; ++i)
		visited[i] = false;
	int count = movingcountCore(threshold,rows,cols,0,0,visited);
	delete[] visited;
	visited = NULL;
	return count;
}
//斐波尼茨序列 三种解法 非课本递归算法
//自顶向下的备忘录法
int fib(int *array, int n)
{
	if (array == NULL)
		return -1;
	if (array[n] != -1)
		return array[n];
	if (n <= 2)
		array[n] = 1;
	else
		array[n] = fib(array, n - 1) + fib(array, n - 2);
	return array[n];
}
int Fibonacci(int n)
{
	if (n <= 0)
		return -1;
	int *memory = new int[n+1];
	for (int i = 0; i < n + 1; i++)
		memory[i] = -1;
	return fib(memory, n);
}
//自底向上的动态规划
int Fibonacci_1(int n)
{
	if (n <= 0)
		return -1;
	int *memory = new int[n + 1];
	for (int i = 0; i < n + 1; i++)
		memory[i] = -1;
	memory[0] = 1;
	memory[1] = 1;
	if (n >= 2)
	{
		for (int i = 2; i <= n; i++)
			memory[i] = memory[i - 1] + memory[i - 2];
	}
	return memory[n];
}
//动态规划题目
int cut(int p[],int n)
{
	if (p == NULL||n <= 0)
		return -1;
	int q = INT_MIN;
	for (int i = 1; i <= n; i++)
	{
		q = max(q, p[i - 1] + cut(p, n - i));
	}
	return q;
}
//自底向上和备忘录自顶向下的没有很大差异; 线性模型
//小朋友过桥的问题
//线性模型;
//有时贪心算法不太对 花费时间不太对,因为只用分析最后一次的用时就行;
//so opt[i]=min{opt[i-1]+a[1]+a[i],opt[i-2]+a[1]+2*a[2]+a[i]} 分析最终的状态
int solution1(int p[],int n)
{
	if (p == NULL||n <= 0)
		return -1;
	int *result_array = new int[n+1];
	for (int i = 0; i <= n; i++)
		result_array[i] = 0;
	if(n==1)
		result_array[1] = p[1];
	if (n==2)
		result_array[2] = p[2];
	if (n>2)
	{
		for (int i = 2; i <= n; i++)
			result_array[i] = min((solution1(p, i - 1) + p[1] + p[i]),(solution1(p, i - 2) + 2 * p[2] + p[1] + p[i]));
	}
	return result_array[n];
}
//0-1背包问题 有没有这是个问题
//f[i][v]=max(f[i-1][v],f[i-1][v-c[i]]+w[i])

//没有理解其中一种 就贴上最为简单的解决办法吧
const int m_max = 1001;//m表示一共有多少种物品
const int n_max = 101;
int m = 0, n = 0;
int w[m_max], c[m_max];
int f[n_max];//n_max表示背包的总重量 m_max表示背包 每种物品的重量
int solution(int w[],int c[],int f[],int m,int n)
{
	for (int i = 1; i <= m;i++)
	for (int v = n; v >= w[i]; v--)
		f[v] = max(f[v - w[i]] + c[i], f[v]);
	return f[n];
}
//完全背包问题
int solution1(int w[], int c[], int f[], int m, int n)
{
	for (int i = 1; i <= m; i++)
	for (int v = w[i]; v <= n; v++)
		f[v] = max(f[v - w[i]] + c[i], f[v]);
	return f[n]; 
}
//剑指offer的链表题 
//从尾到头打印链表
struct ListNode{
	ListNode *next;
	int val;
	ListNode(int x) :val(x), next(NULL){}
};
class solution{
public:
	vector<int> printListFromTailToHead(ListNode *head)
	{
		stack<int> nodes;
		vector<int> list_nodes;
		ListNode *node = head;
		if (head == NULL)
			return list_nodes;
		while (node!=NULL)
		{
			nodes.push(node->val);
			node = node->next;
		}
		while(!nodes.empty())
		{
			int value = nodes.top();
			list_nodes.push_back(value);
			nodes.pop();
		}
		return list_nodes;
	}
};
//链表倒数第k个节点 两个节点 一个快的节点和慢的节点
class solution1{
public:
	ListNode* FindKthToTail(ListNode *pListHead, unsigned int k)
	{
		if (pListHead == NULL || k == 0)
			return NULL;
		ListNode *fast_node = pListHead;
		ListNode *slow_node = pListHead;
		//这块有个bug  写完修正下;链表的长度小于k的情况没有考虑;
		for (int i = 0; i < k - 1; i++)
		{
			if (fast_node->next!=NULL)
				fast_node = fast_node->next;
			return NULL;
		}
		while (fast_node->next!= NULL)
		{
			fast_node = fast_node->next;
			slow_node = slow_node->next;
		}
		return slow_node;
	}
};
//反转链表； 方法 使用三个节点 重点考虑边界情况 这个题一直半解
class solution2{
public:
	ListNode *reverseList(ListNode *pListHead)
	{
		if (pListHead == NULL)
			return NULL;
		ListNode *pCurrentNode = pListHead;
		ListNode *pPreNode = NULL;
		ListNode *pNewHead = NULL;
		while (pCurrentNode != NULL)
		{
			ListNode *pNextNode = pCurrentNode->next;
			if (pNextNode == NULL)
				pNewHead = pCurrentNode;
			pPreNode = pCurrentNode->next;
			pPreNode = pCurrentNode;
			pCurrentNode = pNextNode;
		}
		return pNewHead;
	}
};
//合并链表  记住使用递归;
//递归的性质  
class solution3{
public:
	ListNode *Merge(ListNode *pHead1, ListNode *pHead2)
	{
		if (pHead1 == NULL)
			return pHead2;
		else if (pHead2 == NULL)
			return pHead1;
		ListNode *pMergeHead=NULL;
		if (pHead1->val < pHead2->val)
		{
			pMergeHead = pHead1;
			pMergeHead->next = Merge(pHead1->next,pHead2);
		}
		else{
			pMergeHead = pHead2;
			pMergeHead->next = Merge(pHead1,pHead2->next);
		}
		return pMergeHead;
	}
};
struct RandomListNode{
	int label;
	RandomListNode *next, *random;
	RandomListNode(int x) :label(x), next(NULL), random(NULL){}
};
//复杂链表的复制
//输入一个复杂链表  最简单的方法就是画图，节点比较复杂,不画图有点懵。
class solution4{
public:
	//第一步复制链表
	void CloneNodes(RandomListNode *pHead)
	{
		RandomListNode *pNode = pHead;
		while (pNode != NULL)
		{
			int value = pNode->label;
			RandomListNode *pCloneNode = new RandomListNode(value);
			pCloneNode->next = pNode->next;
			pCloneNode->random = NULL;
			pNode->next = pCloneNode;
			pNode = pNode->next;
		}
	}
	//处理第一步的复制链表 处理复杂指针random
	void ConnectSiblingNodes(RandomListNode *pHead)
	{
		RandomListNode *pNode = pHead;
		while (pNode != NULL)
		{
			RandomListNode *pCloneNode = pNode->next;
			if (pCloneNode == NULL)
				return;
			if (pNode->random != NULL)
				pCloneNode->random = pNode->random->next;
			else
				pCloneNode->random = NULL;
			pNode = pCloneNode->next;
		}
	}
	//拆分复杂链表；
	RandomListNode *ReconnectList(RandomListNode *pHead)
	{
		RandomListNode *pNode = pHead;
		RandomListNode *pCloneHead = NULL;
		RandomListNode *pCloneNode = NULL;
		if (pHead->next!= NULL)
		{
			pCloneHead = pCloneNode = pNode->next;
			pNode->next = pCloneNode->next;
			pNode = pNode->next;
		}
		while (pNode != NULL)
		{
			pCloneNode->next = pNode->next;
			pCloneNode=pCloneNode->next;
			pNode->next = pCloneNode->next;
			pNode = pNode->next;
		}
		return pCloneHead;
	}
};
//两个链表的公共节点。有两种解法
//方案一:就是将两个链表 一个pHead1和pHead2 比如其中一个:pHead1在前 pHead2在后 另外一个:pHead2在前 pHead1在后
//方案二:对列表进行截取 变成同一长度;然后进行比较。代码为方案二.
class solution5{
public:
	ListNode *FindFirstCommonNode(ListNode *pHead1,ListNode *pHead2)
	{
		if (pHead1 == NULL || pHead2 == NULL)
			return NULL;
		int length1 = getListLength(pHead1);
		int length2 = getListLength(pHead2);
		int lengthDif = 0;
		ListNode *pHeadLong = NULL;
		ListNode *pHeadShort = NULL;
		if (length1 < length2)
		{
			pHeadLong = pHead2;
			pHeadShort = pHead1;
			lengthDif = length2 - length1;
		}
		else{
			pHeadLong = pHead1;
			pHeadShort = pHead2;
			lengthDif = length1 - length2;
		}
		for (int i = 0; i < lengthDif; i++)
			pHeadLong = pHeadLong->next;
		while (pHeadLong!=NULL&&pHeadShort!=NULL)
		{
			if (pHeadLong->val == pHeadShort->val)
				break;
			pHeadLong = pHeadLong->next;
			pHeadShort = pHeadShort->next;
		}
		return pHeadLong;
	}
private:
	int getListLength(ListNode *pHead)
	{
		if (pHead == NULL)
			return 0;
		ListNode *pNode = pHead;
		int list_length = 0;
		while (pNode!=NULL)
		{
			list_length++;
			pNode = pNode->next;
		}
		return list_length;
	}
};
//链表中的环的入口节点
//这个分为三个部分;
//1.一个快节点，一个慢节点 同时出发 找到环中的节点
//让这个环的的节点继续向前 记录环中节点的个数
//知道环的节点的个数 建立一个快节点 和一个慢节点 快节点先走一定的步数。
class solution6{
public:
	ListNode *EntryNodeOfLoop(ListNode *pHead)
	{
		if (pHead == NULL)
			return NULL;
		//判断链表有环 所以没有判空的安全性考虑。
		ListNode *MeetingNode = FindMeetingNode(pHead);
		int count = 1;
		ListNode *pNode = MeetingNode->next;
		while (pNode != MeetingNode)
		{
			pNode = pNode->next;
			count++;
		}
		ListNode *fastNode = pHead;
		for (int i = 0; i < count; i++)
			fastNode = fastNode->next;
		ListNode *slowNode = pHead;
		ListNode *resultNode = NULL;
		while (slowNode != fastNode)
		{
			fastNode = fastNode->next;
			slowNode = slowNode->next;
		}
	}
private:
	ListNode *FindMeetingNode(ListNode *pHead)
	{
		if (pHead == NULL)
			return NULL;
		ListNode *fastNode = pHead;
		ListNode *slowNode = pHead;
		ListNode *MeetingNode = NULL;
		while (fastNode->next!= NULL)
		{
			if (fastNode == slowNode)
				MeetingNode = fastNode;
			fastNode = fastNode->next;
			if (fastNode->next == NULL)
				break;
			fastNode = fastNode->next;
			slowNode = slowNode->next;
		}
		return MeetingNode;
	}
};
//删除链表中的重复的节点
//分为两种情况 如果头结点为重复节点 就将头结点更新
class solution7{
public:
	ListNode *deleteDuplication(ListNode *pHead)
	{
		if (pHead == NULL)
			return NULL;
		ListNode *pPreNode = NULL;
		ListNode *pCurNode = pHead;
		ListNode *pNext = NULL;
		while (pCurNode!= NULL)
		{
			if (pCurNode->next != NULL&&pCurNode->val == pCurNode->next->val)
				pNext = pCurNode->next;
			while (pNext->next != NULL&&pNext->next->val == pCurNode->val)
				pNext = pNext->next;
			if (pCurNode == pHead)
			{
				pHead = pNext->next;
			}
			else{
				pPreNode->next = pNext->next;
			}
			pCurNode = pNext->next;
		}
		return pHead; 
	}
};
//二叉树部分
//重建二叉树;前序遍历和中序遍历
struct TreeNode{
	int val;
	TreeNode *leftNode;
	TreeNode *rightNode;
	TreeNode(int x) :val(x), leftNode(NULL), rightNode(NULL){}
};
class solution8{
public:
	TreeNode *reConstructBinaryTree(vector<int> pre, vector<int> vin)
	{
		if (pre.empty() || vin.empty())
			return NULL;
		TreeNode *rootNode = new TreeNode(pre[0]);
		//找到root的索引;  切片是开空间.
		vector<int> pre_left, pre_right, vin_left, vin_right;
		int root = 0;
		for (root = 0; root < pre.size(); root++)
		{
			if (pre[0] == vin[root])
				break;
		}
		if (root == pre.size())
			return NULL;
		for (int i = 0; i < root; i++)
		{
			pre_left.push_back(pre[i+1]);
			vin_left.push_back(vin[i]);
		}
		for (int j = root + 1; j < pre.size(); j++)
		{
			vin_right.push_back(vin[j]);
			pre_right.push_back(pre[j]);
		}
		rootNode->leftNode = reConstructBinaryTree(pre_left,vin_left);
		rootNode->rightNode = reConstructBinaryTree(pre_right,vin_right);
		return rootNode;
	}
};
//树的子树 两步 第一步判断  是不是子树 第二部遍历大的二叉树
class solution9{
public:
	bool HasSubTree(TreeNode *pRoot1, TreeNode *pRoot2)
	{
		bool result = false;
		if (pRoot1 == NULL||pRoot2 == NULL)
			return false;
		if (pRoot1->val == pRoot2->val)
			result = IsSubTree(pRoot1, pRoot2);
		if (!result&&pRoot1->leftNode!=NULL)
			result = HasSubTree(pRoot1->leftNode, pRoot2);
		if (!result&&pRoot1->rightNode != NULL)
			result = HasSubTree(pRoot1->rightNode, pRoot2);
		return result;
	}
private:
	bool IsSubTree(TreeNode *pRoot1,TreeNode *pRoot2)
	{
		if (pRoot2 == NULL)
			return true;
		if (pRoot1 == NULL)
			return false;
		if (pRoot1->val != pRoot2->val)
			return false;
		return IsSubTree(pRoot1->leftNode, pRoot2->leftNode) && IsSubTree(pRoot1->rightNode, pRoot2->rightNode);
	}
};
//二叉树的镜像;
class solution10{
public:
	void Mirror(TreeNode *pRoot)
	{
		if (pRoot == NULL)
			return;
		if (pRoot->leftNode!=NULL||pRoot->rightNode!=NULL)
		{
			TreeNode *temp = pRoot->leftNode;
			pRoot->leftNode = pRoot->rightNode;
			pRoot->rightNode = temp;
		}
		if (pRoot->leftNode != NULL)
			Mirror(pRoot->leftNode);
		if (pRoot->rightNode != NULL)
			Mirror(pRoot->rightNode);
	}
};
//从上往下打印二叉树,记住有个while循环 这里面没有递归 刚刚想的是有递归的。又叫层次遍历二叉树.
class solution11{
public:
	vector<int> printFromTopToBottom(TreeNode *pRoot)
	{
		queue<TreeNode*> nodes;
		TreeNode *pNode;
		nodes.push(pRoot);
		while (!nodes.empty())
		{
			pNode = nodes.front();
			result.push_back(pNode->val);
			if (pNode->leftNode != NULL)
				nodes.push(pNode->leftNode);
			if (pNode->rightNode != NULL)
				nodes.push(pNode->rightNode);
			nodes.pop();
		}
		return result;
	}
private:
	vector<int> result;
};
//二叉树中和为某一值的路径 这道题的核心--回溯法
//满足的条件:1.root的值 是否和tmp相同 2.左子树是否为空 3.右子树是否为空  递归算法里面一般没有while或者for
class solution25{
public:
	vector<vector<int>> FindPath(TreeNode *root,int expectedNumber)
	{
		if (root == NULL)
			return result;
		tmp.push_back(root->val);
		if (root->val - expectedNumber == 0 && root->leftNode == NULL&&root->rightNode == NULL)
			result.push_back(tmp);
		if (root->leftNode!=NULL)
			FindPath(root->leftNode,expectedNumber-root->val);
		if (root->rightNode!=NULL)
			FindPath(root->rightNode,expectedNumber-root->val);
		tmp.pop_back();
		return result;
	}
private:
	vector<vector<int>> result;
	vector<int> tmp;
};
//二叉树的深度  使用DFS或者使用层次遍历的BFS; queue使用的是
class solution12{
public:
	int  TreeDepth_DFS(TreeNode *root)
	{
		if (root == NULL)
			return 0;
		int left = 0, right = 0;
		if (root->leftNode != NULL)
			left = TreeDepth_DFS(root->leftNode);
		if (root->rightNode != NULL)
			right = TreeDepth_DFS(root->rightNode);
		return (left > right) ? (left + 1) : (right + 1);
	}
	int TreeDepth_BFS(TreeNode *root)
	{
		if (root == NULL)
			return 0;
		TreeNode *pNode = NULL;
		queue<TreeNode *> nodes;
		nodes.push(root);
		int depth = 0;
		while (!nodes.empty())
		{
			int size = nodes.size();
			depth++;
			for (int i = 0; i < size; i++)
			{
				pNode = nodes.front();
				if (pNode->leftNode != NULL)
					nodes.push(pNode->leftNode);
				if (pNode->rightNode != NULL)
					nodes.push(pNode->rightNode);
				nodes.pop();
			}
		}
	}
};
//平衡二叉树
//平衡二叉树的定义:所谓的平衡:任意节点的左右子树的高度差不超过一.
//bool 类型判断
class solution13{
public:
	bool IsBalanceTree(TreeNode *root)
	{
		if (root == NULL)
			return false;
		int left = 0,right=0;
		if (root->leftNode!=NULL)
			left = TreeDepth(root->leftNode);
		if (root->rightNode != NULL)
			right = TreeDepth(root->rightNode);
		int diff = left - right;
		if (diff > 1 || diff < -1)
			return false;
		return IsBalanceTree(root->leftNode) && IsBalanceTree(root->rightNode);
	}
private:
	int TreeDepth(TreeNode *root)
	{
		if (root == NULL)
			return 0;
		int left = 0, right = 0;
		if (root->leftNode != NULL)
			left = TreeDepth(root->leftNode);
		if (root->rightNode != NULL)
			right = TreeDepth(root->rightNode);
		return (left > right) ? (left + 1) : (right + 1);
	}
};
//对称的二叉树
class solution14{
public:
	bool Symmetrical(TreeNode *root)
	{
		if (root == NULL)
			return false;
		return isSymmetricalCor(root, root);
	}
private:
	//bool 类型将递归写在后面
	bool isSymmetricalCor(TreeNode *pRoot1, TreeNode *pRoot2)
	{
		if (pRoot1 == NULL&&pRoot2 == NULL)
			return true;
		if (pRoot1 == NULL || pRoot2 == NULL)
			return false;
		if (pRoot1->val != pRoot2->val)
			return false;
		return isSymmetricalCor(pRoot1->leftNode, pRoot2->rightNode) && isSymmetricalCor(pRoot1->rightNode,pRoot2->leftNode);
	}
};
//判断是不是二叉搜索树
class solution15{
public:
	bool VerifySequenceOfBST(vector<int> sequence)
	{
		return bst(sequence, 0, sequence.size()-1);
	}	
private:
	bool bst(vector<int> &sequence, int begin, int end)
	{
		if (sequence.empty()||begin>end)
			return false;
		int rootVal = sequence[end];
		//判断左子树是否为二叉搜索树
		int i = begin;
		for (; i < end; i++)
		{
			if (rootVal<sequence[i])
				break;
		}
		for (int j = i; j < end; j++)
		{
			if (rootVal>sequence[j])
			{
				return false;
			}
		}
		bool left = true,right=true;
		if (i > begin)
			left = bst(sequence,begin,i-1);
		if (i < end - 1)
			right = bst(sequence,i,end-1);
		return left&&right;
	}
};
//二叉搜索树中的第k个节点
class solution16{
public:
	TreeNode *KthNodeOfTree(TreeNode *root, int k)
	{
		if (root == NULL)
			return NULL;
		return KthNodeOfTreeCore(root,k);
	}
private:
	TreeNode *KthNodeOfTreeCore(TreeNode *root,int k)
	{
		TreeNode *pNode;
		if (root == NULL)
			return pNode;
		if (root->leftNode != NULL)
			pNode = KthNodeOfTreeCore(root->leftNode, k);
		if (pNode == NULL)
		{
			if (k == 1)
				pNode = root;
			k--;
		}
		if (pNode->rightNode != NULL)
			pNode = KthNodeOfTreeCore(root->rightNode, k);
		return pNode;
	}
};
//字符串
//c语言用char*指针作为字符串，读取字符串需要一个特殊字符来表示指针的结束位置。
//旋转空格 就不写了 旋转数组的最小数字
class solution17{
public:
	int minNumberInRotateArray(vector<int> &rotateArray)
	{
		int size = rotateArray.size();
		//使用二分查找法;
		if (size == 0)
			return 0;
		int left = 0;
		int right = size - 1;
		int mid = 0;
		while (right > left&&rotateArray[left]>rotateArray[right])
		{
			if (right - left == 1)
			{
				mid = right;
				break;
			}
			int mid = (left + right) >> 2;
			if (rotateArray[mid] >= rotateArray[right])
				left = mid;
			else
				right = mid;
			//特殊情况
			if (rotateArray[mid] == rotateArray[right] && rotateArray[mid] == rotateArray[right])
				return MinInOrder(rotateArray, left, right);
		}
		return rotateArray[mid];
	}
private:
	int MinInOrder(vector<int> &num,int left,int right)
	{
		if (num.empty())
			return 0;
		int min = INT_MAX;
		for (int i = left; i <= right; i++)
		{
			if (num[i] < min)
				min = num[i];
		}
		return min;
	}
};
//字符串的排列  sort(start,end,排序方法)；
class solution18{
public:
	vector<string> Permutation(string str)
	{
		if (str.length() == 0)
			return result;
		int begin = 0;
		result = PermutationCore(str, 0);
		sort(result.begin(), result.end());
		return result;
	}
private:
	void swap(string &str,int i,int j)
	{
		if (str.length() == 0)
			return;
		char temp = str[i];
		str[i] = str[j];
		str[j] = temp;
	}
	vector<string> PermutationCore(string str,int begin)
	{
		if (str.length() == begin)
		{
			result.push_back(str);
			return result;
		}
			
		else{
			for (int i = begin; i < str.length(); i++)
			{
				swap(str, begin, i);
				PermutationCore(str, begin + 1);
				swap(str, begin, i);
			}
		}
	}
	vector<string> result;
};
//第一个只出现一次的字符
class solution19{
public:
	//是否包括大小写;
	char FirstNotRepeatKey(char *pStr)
	{
		if (pStr == NULL)
			return '\0';
		const int HashTableSize = 256;
		int* HashTable = new int[HashTableSize];
		for (int i = 0; i < HashTableSize; i++)
			HashTable[i] = 0;
		char *pHashKey = pStr;
		for (; *pHashKey != '\0'; pStr++)
			HashTable[*pHashKey]++;
		pHashKey = pStr;
		for (; *pHashKey != '\0'; pStr++)
		{
			if (HashTable[*pHashKey] == 1)
				return *pHashKey;
		}
		return '\0';
	}
};
//左旋转字符串 这里有两种方法 先实现比较简单的
class solution20{
public:
	string LeftRotateString(string &str, int n)
	{
		string result = str;
		int length = str.length();
		if (length == 0)
			return NULL;
		if (n>0 && n< length)
		{
			int first_begin = 0, first_end = n - 1;
			int second_begin = n, second_end = length - 1;
			ReverseString(str, first_begin, first_end);
			ReverseString(str, second_begin, second_end);
			ReverseString(str,first_begin,second_end);
		}
		return result;
	}
private:
	void ReverseString(string &str,int start,int end)
	{
		while (start < end)
		{
			char temp = str[start];
			str[start] = str[end];
			str[end] = temp;

			start++;
			end--;
		}
	}
};
//翻转单词顺序序列
//b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.
//所以a[i:j : 1]相当于a[i:j]
//当s<0时，i缺省时，默认为 - 1. j缺省时，默认为 - len(a) - 1
//所以a[:: - 1]相当于 a[-1:-len(a) - 1 : -1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序
class solution21{
public:
	string ReverseSetence(string str)
	{
		string result = str;
		int length = result.length();
		if (length == 0)
			return NULL;
		int begin = 0, end = 0;
		for (int i = 0; i < length; i++)
		{
			if (str[i] == ' ')
			{
				end = i - 1;
				ReverseString(result, begin, end);
				begin = i + 1;
			}
		}
		ReverseString(result, 0, length - 1);
		return result;
	}
private:
	void ReverseString(string &str, int start, int end)
	{
		while (start < end)
		{
			char temp = str[start];
			str[start] = str[end];
			str[end] = temp;

			start++;
			end--;
		}
	}
};

//排序算法 快速排序;
class solution22{
public:
	void QuickSort(vector<int> &list,int left,int right)
	{
		while (left < right)
		{
			int base = QuickSortCore(list, left, right);
			QuickSort(list, left, base - 1);
			QuickSort(list, base + 1, right);
		}
	}
private:
	//先写个代码 返回快排的下标
	int QuickSortCore(vector<int> &list,int left,int right)
	{
		if (list.empty())
			return -1;
		int base = list[0];
		while (left < right)
		{
			while (left < right&&list[right] > base)
				right--;
			list[left] = list[right];
			while (left < right&&list[left] < base)
				left++;
			list[right] = list[left];
		}
		list[left] = base;
		return left;
	}
};

//冒泡排序算法 简洁版 如果不发生交换 直接停止循环。
class solution23{
public:
	void bubbleSort(vector<int> &list)
	{
		if (list.empty())
			return;
		//最后一遍不用排; 
		for (int i = 0; i < list.size()-1; i++)
		{
			bool changed_flag = false;
			for (int j = 0; j < list.size()-i-1; j++)
			{
				if (list[j] < list[j + 1])
				{
					swap(list, i, j);
					changed_flag = true;
				}
			}
			if (!changed_flag)
				break;
		}
	}
private:
	void swap(vector<int> &list, int i, int j)
	{
		if (list.empty())
			return;
		int temp = list[i];
		list[i] = list[j];
		list[j] = temp;
	}
};
//直接插入排序算法 典型的直接插入排序，每次将一个新数据插入到有序队列中的合适位置里。
class Solution24{
public:
	vector<int> InsertSort(vector<int> &list)
	{
		vector<int> result;
		if (list.empty()){
			return result;
		}
		result = list;
		// 第1个数肯定是有序的，从第2个数开始遍历，依次插入有序序列
		for (int i = 1; i < result.size(); i++){
			// 取出第i个数，和前i-1个数比较后，插入合适位置
			int temp = result[i];
			// 因为前i-1个数都是从小到大的有序序列，所以只要当前比较的数(list[j])比temp大，就把这个数后移一位
			int j = i - 1;
			for (j; j >= 0 && result[j] > temp; j--){
				result[j + 1] = result[j];
			}
			result[j + 1] = temp;
		}
		return result;
	}
private:
	void swap(vector<int> &list, int i, int j)
	{
		if (list.empty())
			return;
		int temp = list[i];
		list[i] = list[j];
		list[j] = temp;
	}
};
int _tmain(int argc, _TCHAR* argv[])
{
	int arr[] = { 6, 4, 8, 9, 2, 3, 1 };
	vector<int> test(arr, arr + sizeof(arr) / sizeof(arr[0]));
	cout << "排序前" << endl;
	for (int i = 0; i < test.size(); i++){
		cout << test[i] << " ";
	}
	cout << endl;
	vector<int> result;
	Solution24 solution;
	result = solution.InsertSort(test);
	cout << "排序后" << endl;
	for (int i = 0; i < result.size(); i++){
		cout << result[i] << " ";
	}
	cout << endl;
	getchar();
	return 0;
}

