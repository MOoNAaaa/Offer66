#include <iostream>
#include <vector>
#include <stack>
#include <queue>
//#include <bits/unordered_map.h>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <map>

using namespace std;
struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x), left(NULL), right(NULL){
    }
};
struct RandomListNode{
    int label;
    RandomListNode* next;
    RandomListNode* random;
    RandomListNode(int x):label(x), next(NULL), random(NULL){}
};
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
            val(x), next(NULL) {
    }
};
class Solution{
public:
    /*
     * 重建二叉树
     */
    /*
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     */
    TreeNode* reConstructBinaryTree(vector<int> preArr, vector<int> inArr){
        if(preArr.size() ==0 || inArr.size() ==0 || preArr.size() != inArr.size())
            return NULL;
        return reConstructBinaryTree(preArr, inArr, 0, preArr.size() - 1, 0, inArr.size() - 1);
    }
    TreeNode* reConstructBinaryTree(vector<int> preArr, vector<int> inArr, int preStart, int preEnd, int inStart, int inEnd){
        if(preEnd - preStart < 0 || inEnd - inStart < 0)
            return NULL;
        TreeNode* pRoot = new TreeNode(preArr[preStart]);
        int curIndexInInArr = findIndexOfVal(inArr, inStart, inEnd, preArr[preStart]);
        int leftNums = curIndexInInArr - inStart;   // 以cur为根的左子树结点个数
        int rightNums = inEnd - curIndexInInArr;
        if(leftNums > 0){
            pRoot->left = reConstructBinaryTree(preArr, inArr, preStart + 1, preStart + leftNums, inStart, inStart + leftNums - 1);
        }
        if(rightNums > 0){
            pRoot->right = reConstructBinaryTree(preArr, inArr, preStart + leftNums + 1, preEnd, inStart + leftNums + 1, inEnd);
        }
        return pRoot;
    }
    int findIndexOfVal(vector<int> arr, int start, int end, int val){    // 返回某值在数组中的下标, 没找到返回 -1
        if(end - start >= 0){
            for(int i=start; i<=end; i++){
                if(arr[i] == val){
                    return i;
                }
            }
        }
        return -1;
    }

    /*
     * 用两个栈实现队列
     */
    /*
     * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
     * 思路：pop: 只从 stack2 中弹出元素，若 stack2 为空，将 stack1 中的元素全部倒入 stack2
     *      push： 直接将元素压入 stack1
     */
    int pop(){
        int res;
        if(stack2.empty()){
            while (!stack1.empty()){
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        res = stack2.top();
        stack2.pop();
        return res;
    }
    void push(int x){
        stack1.push(x);
    }
    /*
     * 旋转数组的最小数字
     */
    /*
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     * 思路：设置两个指针，一个指针指向前一个数组中最后一个元素，一个指针指向后一个数组中第一个元素，当两个指针相遇或相邻时即能找到最小值
     *       当 arr[mid] > arr[right] , mid 指向前一半数组中的元素
     *       当 arr[mid] <= arr[right]， mid 指向后一半数组中的元素
     *       特殊情况： arr[mid] == arr[left] == arr[right] 只能顺序查找数组最小值
     */
    /*
     * 迭代方法
     */
    int minNumberInRotateArray(vector<int> rotateArr){
        if(rotateArr.size() == 0)
            return 0;
        int left = 0, right = rotateArr.size() - 1;
        while (right - left > 1){
            int mid = (left + right) / 2;
            if(rotateArr[mid] == rotateArr[left] && rotateArr[left] == rotateArr[right])
                return findMinNumberBySeq(rotateArr, left, right);
            if(rotateArr[mid] > rotateArr[right]){
                left = mid + 1;
            } else{     // if(rotateArr[mid] <= rotateArr[right])
                right = mid;
            }
        }
        if(right - left == 1 || right == left){
            return rotateArr[left] < rotateArr[right] ? rotateArr[left] : rotateArr[right];
        }
    }
    /*
     * 递归方法
     */
    int minNumberInRotateArray2(vector<int> rotateArr){
        if(rotateArr.size() == 0 )
            return 0;
        return minNumberInRotateArray2(rotateArr, 0, rotateArr.size() - 1);
    }
    int minNumberInRotateArray2(vector<int> rotateArr, int left, int right){
        if(right - left > 1){
            int mid = (right + left) / 2;
            if(rotateArr[mid] == rotateArr[left] && rotateArr[left] == rotateArr[right]){
                return findMinNumberBySeq(rotateArr, left, right);
            }
            if(rotateArr[mid] > rotateArr[right]){
                return minNumberInRotateArray2(rotateArr, mid + 1, right);
            } else if(rotateArr[mid] <= rotateArr[right]){
                return minNumberInRotateArray2(rotateArr, left, mid);
            }
        }
        if(right == left)
            return rotateArr[left];
        if(right - left == 1)
            return rotateArr[right] < rotateArr[left] ? rotateArr[right] : rotateArr[left];
    }
    int findMinNumberBySeq(vector<int> arr, int left, int right){
        int res = 0;
        for(int i=left; i<=right; i++){
            if(arr[i] < res){
                res = arr[i];
            }
        }
        return res;
    }
    /*
     * 数值的整数次方
     */
    double Power(double base, int exponent) {
        if(exponent == 0)
            return 1;
        if(base == 0)
            return 0;
        bool isExponentNeg = false;
        if(exponent < 0){
            isExponentNeg = true;
            exponent = -exponent;
        }
        double res = base;
        bool isOdd = false;
        if(exponent % 2 == 1)
            isOdd = true;
        while(exponent > 1 && exponent / 2){
            res *= res;
            exponent = exponent / 2;
        }
        if(isOdd)
            res = res * base;
        if(isExponentNeg)
            res = 1.0 / res;
        return res;
    }
    /*
     * 树的子结构
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     * 思路：两个函数，  HasSubtree 判断B是不是A的子结构
     *                  isSubtree 判断B是不是A的同根子结构
     */
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)     // 树 pRoot1 中是否有子树 pRoot2
    {
        if(pRoot1 && pRoot2){
            if(isSubtree(pRoot1, pRoot2) == false){
                return HasSubtree(pRoot1->left, pRoot2) ||
                       HasSubtree(pRoot1->right, pRoot2);
            }else{
                return true;;
            }
        }
        return false;
    }
    bool isSubtree(TreeNode* pRoot1, TreeNode* pRoot2){ // 树 pRoot2 是否是 pRoot1 以 pRoot1 为根的子树
        if(pRoot2 == NULL)
            return true;
        if(pRoot1 == NULL)
            return false;
        if(pRoot1->val == pRoot2->val)
            return isSubtree(pRoot1->left, pRoot2->left) &&
                   isSubtree(pRoot1->right, pRoot2->right);
        return false;
    }
    /*
     * 二叉树的镜像
     * 操作给定的二叉树，将其变换为源二叉树的镜像。
     */
    void Mirror(TreeNode* pRoot){
        if(pRoot != NULL){
            SwapNode(pRoot->left, pRoot->right);
            if(pRoot->left)
                Mirror(pRoot->left);
            if(pRoot->right)
                Mirror(pRoot->right);
        }
    }
    void SwapNode(TreeNode* &p1, TreeNode* &p2){
        TreeNode* temp = p1;
        p1 = p2;
        p2 = temp;
    }
    /*
     * 顺时针打印矩阵
     */
    vector<int> printMatrix(vector<vector<int >> matrix){
        vector<int> res;
        if(matrix.size() == 0)
            return res;
        int row1 = 0, row2 = matrix.size() - 1;
        int col1 = 0, col2 = matrix[0].size() - 1;
        while (row2 - row1 >= 0 && col2 - col1 >= 0){
            printMatrix(matrix, res, row1++, row2--, col1++, col2--);
        }
        return res;
    }
    void printMatrix(vector<vector<int >> matrix, vector<int> &res, int row1, int row2, int col1, int col2){
        if(row2 - row1 >= 0 && col2 - col1 >= 0){
            // 只有一行
            if(row1 == row2){
                for(int i=col1; i<=col2; i++){
                    res.push_back(matrix[row1][i]);;
                }
                return;
            }
            // 只有一列
            if(col1 == col2){
                for(int i=row1; i<=row2; i++){
                    res.push_back(matrix[i][col1]);
                }
                return;
            }
            // 从左到右
            for(int i=col1; i<col2; i++){
                res.push_back(matrix[row1][i]);
            }
            // 从上到下
            for(int i=row1; i<row2; i++){
                res.push_back(matrix[i][col2]);
            }
            // 从右到左
            for(int i=col2; i>col1; i--){
                res.push_back(matrix[row2][i]);
            }
            // 从下到上
            for(int i=row2; i>row1; i--){
                res.push_back(matrix[i][col1]);
            }
        }
    }
    /*
     * 包含min函数的栈
     * 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
     * 思路：
     */
    void push1(int x){
        if(minStack.empty() || x <= minStack.top()){
            minStack.push(x);
        }
        dataStack.push(x);
    }
    void pop1(){
        if(dataStack.top() == minStack.top()){
            minStack.pop();
        }
        dataStack.pop();
    }
    int top(){
        return dataStack.top();
    }
    int min(){
        return minStack.top();
    }
    /*
     * 栈的压入、弹出序列
     *
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
     * 假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
     * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
     *
     * 思路：使用辅助栈，将pushV的元素一个个压栈，
     * 当遇到栈顶与popV中元素相等时，弹栈，指向popV的下标后移
     */
    bool isPopOrder(vector<int> pushV, vector<int> popV){
        if(pushV.size() != popV.size())
            return false;
        stack<int> s;
        for(int i=0, j=0; i<pushV.size() && j<popV.size(); i++){
            s.push(pushV[i]);
            while (j<popV.size() && !s.empty() && s.top() == popV[j]){ // 保证j不越界，栈不空，栈顶和popV元素相等
                s.pop();
                j++;
            }
        }
        if(s.empty())
            return true;
        return false;
    }
    /*
     * 从上往下打印二叉树（层序打印二叉树，借助队列）
     *
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     */
    vector<int> PrintFromTopToBottom(TreeNode* pRoot){
        vector<int> res;
        if(pRoot == NULL)
            return res;
        queue<TreeNode*> q;
        TreeNode* p;
        q.push(pRoot);
        while (!q.empty()){
            p = q.front();
            q.pop();
            res.push_back(p->val);
            if(p->left)
                q.push(p->left);
            if(p->right)
                q.push(p->right);
        }
        return res;
    }
    /*
     * 二叉搜索树的后序遍历序列 （BST, postOrder）
     *
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
     *
     * 二叉排序树的性质：左子树上所有节点的值均小于它的根节点；右子树上所有节点的值均大于它的根节点。
     * 二叉排序树后序遍历的性质：序列最后一个数字是根节点，序列剩余部分分成两部分，前一部分是左子树，后一部分是右子树。
     */
    bool VerifySquenceOfBST(vector<int> sequence){
        if(sequence.size() == 0)
            return false;
        int rootIndex = sequence.size() - 1;
        int start = 0;
        int end = sequence.size() - 2;
        return IsPostOrderOfBST(sequence, rootIndex, start, end);
    }
    bool IsPostOrderOfBST(vector<int> sequence, int rootIndex, int start, int end){
        if(end >= start){
            int smallSet = start - 1;   // smallset 指向 start 前一个
            int bigSet = end + 1;   // bigSet 指向 end 后一个
            int i = start, j = end;
            while (i <= end && sequence[i] < sequence[rootIndex]){
                smallSet++;
                i++;
            }
            while (j>= start && sequence[j] > sequence[rootIndex]){
                bigSet--;
                j--;
            }
            if(smallSet == bigSet - 1)  // 如果 smallset 和 bigset 是相邻的，通过当前检查，继续检查左右子区间
                return IsPostOrderOfBST(sequence, smallSet, start, smallSet - 1) && IsPostOrderOfBST(sequence, end, bigSet, end-1);
            return false;
        }
        return true;
    }
    /*
     * 二叉树中和为某一值的路径
     *
     * 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
     * 路径定义：从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
     * 每条满足条件的路径都是以根节点开始，叶子结点结束，
     * 如果想得到所有根节点到叶子结点的路径（不一一定满足和为某整数的条件），需要遍历整棵树，还要先遍历根节点，所以采用先序遍历
     * (注意: 在返回值的list中，数组长度大的数组靠前)
     */
    vector<vector<int >> FindPath(TreeNode* pRoot, int expectNumber){
        vector<vector<int >> res;
        if(pRoot == NULL)
            return res;
        vector<int> path;
        int curNumber = 0;
        FindPath(pRoot, res, path, curNumber, expectNumber);
        return res;
    }
    // 深度优先遍历（DFS） 树的深度优先遍历即先根遍历
    void FindPath(TreeNode* pRoot, vector<vector<int >> &res, vector<int> &path, int curNumber, int expectNumber){
        if(pRoot == NULL)
            return;

        path.push_back(pRoot->val);
        curNumber += pRoot->val;
        bool isLeaf = pRoot->left == NULL && pRoot->right == NULL;
        if(isLeaf && curNumber == expectNumber){  // 如果当前节点是叶节点，且满足条件，将该路径加入res（先处理根）
            res.push_back(path);
        }
        if(curNumber < expectNumber ){  // 当满足条件，继续遍历下去（先左后右遍历左右子树）
            if(pRoot->left){
                FindPath(pRoot->left, res, path, curNumber, expectNumber);
            }
            if(pRoot->right){
                FindPath(pRoot->right, res, path, curNumber, expectNumber);
            }
        }
        path.pop_back();    // *移除当前节点*
    }
    /*
     * 复杂链表的复制
     *
     * 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
     * 返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
     *
     * 思路1：使用 hashSet, 时复 O（n）， 空复 O（n）
     * （key, value） = （oldNode, newNode）
     * newNode->val = oldNode->val;
     * newNode->next = oldNode->next;
     * pNew->random = mymap[pOld->random];
     */
//    RandomListNode* Clone(RandomListNode* pHead){
//        if(pHead == NULL)
//            return NULL;
//        unordered_map<RandomListNode*, RandomListNode*> mymap;
//        RandomListNode* resHeadPre = new RandomListNode(0);     // 头结点的前一个结点（为方便编写，浪费一个结点）
//        RandomListNode* pOld = pHead;
//        RandomListNode* pNew = resHeadPre;
//        while (pOld){
//            RandomListNode* node = new RandomListNode(pOld->label);
//            pNew->next = node;
//            pNew = node;
//            mymap.insert(make_pair(pOld, pNew));
//            pOld = pOld->next;
//        }
//        // 复制 random 指针
//        pNew = resHeadPre->next;
//        pOld = pHead;
//        while (pOld){
//            pNew->random = mymap[pOld->random];
//            pNew = pNew->next;
//            pOld = pOld->next;
//        }
//        return resHeadPre->next;
//    }
    /*
     * 复杂链表的复制
     *
     * 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
     * 返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
     *
     * 思路2：不使用 hashSet, 时复 O（n）， 空复 O（1）
     * 例：1->2->3->null
     * ① 1->1'->2->2'->3->3'->null
     * ② 连接random指针：1'->random = 1->random->next
     * ③ 拆分链表
     */
//    RandomListNode* Clone2(RandomListNode* pHead){
//        if(pHead == NULL)
//            return NULL;
//        // 复制结点并设置 next 和 label
//        RandomListNode* p = pHead;
//        RandomListNode* pNext = p->next;
//        while (p){
//            RandomListNode* node = new RandomListNode(p->label);
//            node->next = pNext;
//            p->next = node;
//            p = pNext;
//            if(pNext)
//                pNext = pNext->next;
//        }
//        // 设置 random  11'22'33'null
//        p = pHead;
//        pNext = p->next;
//        while (p){
//            if(p->random){
//                pNext->random = p->random->next;
//            }
//            if(pNext){
//                p = pNext->next;
//            }
//            if(p){
//                pNext = p->next;
//            }
//        }
//        // 拆分链表 11'22'33'null
//        p = pHead;
//        pNext = p->next;
//        RandomListNode* resHead = pNext;
//        while (p){
//            if(pNext){
//                p->next = pNext->next;
//                p = p->next;
//            }
//            if(p){
//                pNext->next = p->next;
//                pNext = pNext->next;
//            }
//        }
//        return resHead;
//    }
    /*
     * 二叉搜索树与双向链表
     *
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
     * 要求不能创建任何新的结点，只能调整树中结点指针的指向。
     *
     * 思路1（递归中序遍历）：二叉搜索树中序遍历结果是升序的，在中序遍历二叉搜索树的过程中，记录当前结点和下一个结点，修改两个结点的指针
     * 1 2 3
     * 1->right = 2;
     * 2->left = 1;
     * 2->right = 3;
     * 3->left = 2;
     */
    TreeNode* BinaryTreeConvertList(TreeNode* pRoot){
        if(pRoot ==NULL)
            return NULL;
        TreeNode* pCur = pRoot;
        TreeNode* preCur = NULL;
        BinaryTreeConvertList(pCur, preCur);
        while (pCur->left){
            pCur = pCur->left;
        }
        return pCur;
    }
    void BinaryTreeConvertList(TreeNode* pCur, TreeNode* &preCur){
        // 第一个用指针的原因：pCur的值不需要改变，只需改变pCur指向的值
        // 第二个用指针引用原因：preCur的值需要改变
        if(pCur == NULL)
            return;
        if(pCur->left){
            BinaryTreeConvertList(pCur->left, preCur);
        }
        pCur->left = preCur;
        if(preCur)
            preCur->right = pCur;
        preCur = pCur;
        if(pCur->right){
            BinaryTreeConvertList(pCur->right, preCur);
        }
    }
    /*
     * 二叉搜索树与双向链表
     *
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
     * 要求不能创建任何新的结点，只能调整树中结点指针的指向。
     *
     * 思路1（非递归中序遍历）：二叉搜索树中序遍历结果是升序的，在中序遍历二叉搜索树的过程中，记录当前结点和下一个结点，修改两个结点的指针
     * 1 2 3
     * 1->right = 2;
     * 2->left = 1;
     * 2->right = 3;
     * 3->left = 2;
     */
    TreeNode* Convert(TreeNode* pRoot){
        if(pRoot == NULL)
            return NULL;
        TreeNode* pre = NULL;
        TreeNode* cur = pRoot;
        stack<TreeNode*> s;
        while (cur || !s.empty()){
            while (cur){
                s.push(cur);
                cur = cur->left;
            }
            if(!s.empty()){
                cur = s.top();
                s.pop();

                // 修改指针
                cur->left = pre;
                if(pre){
                    pre->right = cur;
                }
                pre = cur;
                // 修改指针

                cur = cur->right;
            }
        }
        // 找到链表头， 没必要从尾节点开始找， 可以直接从根节点开始找
        cur = pRoot;
        while (cur->left){
            cur = cur->left;
        }
        return cur;
    }
    /*
     * 字符串的排列
     *
     * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
     * 例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
     * 输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
     *
     * 思路：把一个字符串看成两部分组成：第一部分为第一个字符，第二部分为后面的所有字符。
     * 求整个字符串的排列，可以看出两步：首先求所有可能出现在第一个位置的字符，
     * 即把第一个字符和后面的所有字符交换；然后固定第一个字符，求后面所有字符的排序。
     * 此时仍把后面的字符看成两部分，第一个字符和后面的字符，然后重复上述步骤。（递归）
     */
    vector<string> Permutation(string str){
        vector<string> res;
        int curIndex = 0;
//        char* cur = &str[0];
//        Permutation(str, cur, res);
        Permutation(str, curIndex, res);
        sort(res.begin(), res.end());
        return res;
    }
    /*
     * 方法1：使用指针
     */
    void Permutation(string &str, char* cur, vector<string> &res){
        if(*cur == '\0'){
            res.push_back(str);
            return;
        }
        for(char* i = cur; *i != '\0'; i++){
            if(i != cur && *i == *cur)
                continue;
            char temp = *i;
            *i = *cur;
            *cur = temp;

            Permutation(str, cur + 1, res);
            // 换回来
            temp = *i;
            *i = *cur;
            *cur = temp;
        }
    }
    /*
     * 方法2：使用下标
     */
    void Permutation(string &str, int curIndex, vector<string> &res){
        if(curIndex == str.length()){
            res.push_back(str);
            return;
        }
        for(int i = curIndex; i<str.length() && curIndex < str.length(); i++){
            if(i != curIndex && str[i] == str[curIndex])
                continue;
            char temp = str[i];
            str[i] = str[curIndex];
            str[curIndex] = temp;
            Permutation(str, curIndex + 1, res);
            // 换回来
            temp = str[i];
            str[i] = str[curIndex];
            str[curIndex] = temp;
        }
    }
    /*
     * 数组中出现次数超过一半的数字
     *
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
     * 由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
     */
    int MoreThanHalfNum(vector<int> arr){
        if(arr.size() == 0)
            return 0;
        int res = arr[0];
        int count = 1;
        for(int i=1; i<arr.size(); i++){
            if(arr[i] == res){
                count++;
            }else{
                if(count > 1){
                    count--;
                } else{
                    res = arr[i];
                    count = 1;
                }
            }
        }
        if( count > 1 ){
            return res;
        }
        count = 0;
        for(int i=0; i<arr.size(); i++){    // verify
            if(arr[i] == res)
                count++;
        }
        if(count * 2 > arr.size())
            return res;
        return 0;
    }
    /*
     * 最小的K个数
     *
     * 输入n个整数，找出其中最小的K个数。
     * 例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     *
     * 思路1：partition， 需要改变数组，且找到的最小 k 个数是乱序的
     */
    vector<int> minKNumbers(vector<int> arr, int k){
        vector<int> res;
        if(k > arr.size())
            return res;
        int begin = 0;
        int end = arr.size() - 1;
        int index = Partition(arr, begin, end);
        if(index != k - 1){
            if(index < k - 1){
                index = Partition(arr, index + 1, end);
            } else{
                index = Partition(arr, begin, index - 1);
            }
        }

        for(int i=0; i<k; i++){
            res.push_back(arr[i]);
        }
        return res;
    }
    int Partition(vector<int> &arr, int begin, int end){
        int pivot = arr[begin];
        int i = begin, j = end;
        while (i < j){
            while (i < j && arr[j] >= pivot ){
                j--;
            }
            while (i < j && arr[i] <= pivot){
                i++;
            }
            if(i < j)
                swap(arr[i], arr[j]);
        }
        swap(arr[begin], arr[i]); // 将枢值放在应该放的位置上
        return i;
    }
    /*
     * 最小的K个数
     *
     * 输入n个整数，找出其中最小的K个数。
     * 例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     *
     * 思路2：不改变数组，排序
     * 思路3: 不改变数组，堆
     */
    vector<int> GetLeastNumbers(vector<int> arr, int k){
        vector<int> res;
        if(k > arr.size())
            return res;
        priority_queue<int> minK;   // 建立最小k个数的大根堆
        for(int i=0; i<k; i++){
            minK.push(arr[i]);
        }
        for(int i=k; i<arr.size(); i++){
            if(!minK.empty() && arr[i] < minK.top()){
                minK.pop();
                minK.push(arr[i]);
            }
        }
        while (!minK.empty()){
            res.push_back(minK.top());
            minK.pop();
        }
        return res;
    }
    vector<int> GetLeastNumbers_Solution(vector<int> arr, int k){
        vector<int> res;
        if(k > arr.size())
            return res;
        multiset<int, greater<int>> minK;
        for(int i = 0; i < k; i++){
            minK.insert(arr[i]);
        }
        for (int i = k; i < arr.size(); ++i) {
            if(minK.size() > 0 && arr[i] < *minK.begin()){
                minK.erase(*minK.begin());
                minK.insert(arr[i]);
            }
        }
        multiset<int>::iterator it = minK.begin();
        for (; it != minK.end(); ++it) {
            res.push_back(*it);
        }
        return res;
    }
    /*
     * 使用大根堆对数组进行排序
     * priority_queue<Type, Container, Functional>
     * 如果我们把后面俩个参数缺省的话，优先队列就是大顶堆，队头元素最大。
     */
    vector<int> maxHeapSort(vector<int> arr){
        vector<int> res;
        priority_queue<int> maxheap;
        for(int i=0; i<arr.size(); i++){
            maxheap.push(arr[i]);
        }
        while (!maxheap.empty()){
            res.push_back(maxheap.top());
            maxheap.pop();
        }
        return res;
    }
    /*
     * 使用小根堆对数组进行排序
     *
     * priority_queue<Type, Container, Functional>
     */
    vector<int> minHeapSort(vector<int> arr){
        vector<int> res;
        priority_queue<int, vector<int>, greater<int> > minheap;
        for(int i=0; i<arr.size(); i++){
            minheap.push(arr[i]);
        }
        while (!minheap.empty()){
            res.push_back(minheap.top());
            minheap.pop();
        }
        return res;
    }
    /*
     * STL使用堆方法：
     * 1. 优先队列，默认大顶堆 priority_queue<int>
     *                 小顶堆 priority_queue<int, vector<int>, greater<int> >
     * 2. 利用头文件 <algothrim> 中的函数进行堆操作   （1）make_heap(_First, _Last, _Comp)构造堆：默认是建立最大堆的。对int类型，可以在第三个参数传入greater<int>()得到最小堆。
     *                                              （2）push_heap (_First, _Last)添加元素到堆：先在容器中加入，再调用push_heap()
     *                                              （3）pop_heap(_First, _Last)从堆中移出元素：先调用pop_heap()，再在容器中删除
     *                                              （4）sort_heap(_First, _Last)对整个堆排序：排序之后就不再是一个合法的heap结构了， 容器变成有序数组
     */

    /*
     * 对称的二叉树
     *
     * 请实现一个函数，用来判断一颗二叉树是不是对称的。
     * 注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
     *
     * 思路：先序遍历左右子树
     */
    bool isSymmetrical2(TreeNode* pRoot) {
        if(pRoot == NULL)
            return true;
        return isSymmetrical(pRoot->left, pRoot->right);
    }
    bool isSymmetrical(TreeNode* pRoot1, TreeNode* pRoot2) {
        if(pRoot1 == NULL && pRoot2 == NULL)
            return true;
        if(pRoot1 && pRoot2){
            if(pRoot1->val == pRoot2->val){
                return isSymmetrical(pRoot1->left, pRoot2->right) && isSymmetrical(pRoot1->right, pRoot2->left);
            }else{
                return false;
            }
        }
        return false;
    }

    /*
     * 连续子数组的最大和
     *
     * 例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和(子向量的长度至少是1)
     *
     * 思路：res 记录当前连续子向量最大和； curSum 记录当前连续子向量的和
     *      遍历数组， curSum + arr[i] >= 0 时， 继续使用当前的 curSum， 否则重置 curSum 为 0
     *      过程中一旦发现 curSum > res , 更新 res 的值
     */
    int FindGreatestSumOfSubArray(vector<int> arr){
        if(arr.size() == 0)
            return 0;
        int res = 0;
        int curSum = 0;
        bool hasPos = false;
        for(int i=0; i<arr.size(); i++){
            if(curSum + arr[i] >= 0){
                curSum += arr[i];
            }else{
                curSum = 0;
            }
            if(curSum > res){
                res = curSum;
                hasPos = true;
            }
        }
        if(hasPos == false){ // 如果数组中全是负数，返回最大值
            res = *max_element(arr.begin(), arr.end());
        }
        return res;
    }
    /*
     * 二叉搜索树的第k个结点
     *
     * 给定一棵二叉搜索树，请找出其中的第k小的结点。
     * 例如，（5，3，7，2，4，6，8） 中，按结点数值大小顺序第三小结点的值为4。
     *
     * 思路：中序遍历
     */
    TreeNode* KthNode(TreeNode* pRoot, int k){
        if(pRoot == NULL || k <= 0)
            return NULL;
        TreeNode* p = pRoot;
        stack<TreeNode*> s;
        int cur = 0;
        while (p || !s.empty()){
            while (p){
                s.push(p);
                p = p->left;
            }
            if(!s.empty()){
                p = s.top();
                s.pop();
                cur++;
                if(cur == k){
                    return p;
                }
                p = p->right;
            }
        }
        return NULL;
    }
    /*
     * 把二叉树打印成多行
     *
     * 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
     *
     * 思路：先求出二叉树高度，初始化数组，再层序遍历一行一行放
     */
    vector<vector<int> > PrintBTByLevel(TreeNode* pRoot){
        int height = BTHeight(pRoot);
        vector<vector<int> > res(height);
        if(pRoot == NULL)
            return res;
        TreeNode* p;
        queue<TreeNode*> q;
        q.push(pRoot);
        int curLine = 0;
        while (!q.empty()){
            int curCount = 0, eleSum = q.size();  // curCount: 当前行加入元素计数    eleSum：当前行应该加入元素总数
            while(curCount < eleSum) {
                p = q.front();
                q.pop();
                res[curLine].push_back(p->val);
                curCount++;
                if(p->left){
                    q.push(p->left);
                }
                if(p->right){
                    q.push(p->right);
                }
            }
            curLine++;
        }
        return res;
    }
    int BTHeight(TreeNode* pRoot){
        if(pRoot == NULL)
            return 0;
        return max(BTHeight(pRoot->left), BTHeight(pRoot->right)) + 1;
    }
    /*
     * 把二叉树打印成多行
     *
     * 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
     *
     * 思路：不求二叉树高度，每层先存在一个vector<int>中， 再一行一行放进 vector<vector<int> > 中
     */
    vector<vector<int> > PrintBTByLevel2(TreeNode* pRoot){
        vector<vector<int> > res;
        if(pRoot == NULL)
            return res;
        queue<TreeNode*> q;
        q.push(pRoot);
        while (!q.empty()){
            vector<int> curLine;
            int start = 0, end = q.size();
            while (start < end){
                TreeNode* p = q.front();
                q.pop();
                curLine.push_back(p->val);

                if(p->left){
                    q.push(p->left);
                }
                if(p->right){
                    q.push(p->right);
                }

                start++;

            }
            res.push_back(curLine);
        }
        return res;
    }

    /*
     * 按之字形顺序打印二叉树
     *
     * 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，
     * 第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
     *
     * 思路：
     */
    vector<vector<int> > PrintBTByZhi(TreeNode* pRoot){
        vector<vector<int> > res;
        if(pRoot == NULL)
            return res;
        queue<TreeNode*> q;
        q.push(pRoot);
        bool direction = true; // true: 从左往右， false：从右往左
        while (!q.empty()){
            vector<int> curLine;
            int start = 0, end = q.size();
            while (start < end){
                TreeNode* p = q.front();
                q.pop();
                curLine.push_back(p->val);

                if(p->left){
                    q.push(p->left);
                }
                if(p->right){
                    q.push(p->right);
                }
                start++;
            }
            if(direction){  // true: 从左往右
                res.push_back(curLine);
            }else{          // false：从右往左
                vector<int> temp;
                int size = curLine.size();  // 需要使用 size 保存 vector 的长度，因为后面的操作会修改 vector 的长度
                for(int i=0; i<size; i++){
                    temp.push_back(*(curLine.end()-1));
                    curLine.pop_back();
                }
                res.push_back(temp);
            }
            direction = !direction;
        }
        return res;
    }
    /*
     * 删除链表中重复的结点
     *
     * 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
     * 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
     *
     * 思路：先遍历一遍链表，用 set 记录下有重复值的结点
     *       再遍历一遍链表，如果当前经查询后确认没有重复值，将其加入链表，否则删除
     */

    ListNode* deleteDuplicateNode(ListNode* pHead) {
        if(pHead == NULL)
            return NULL;
        set<int> isDupSet;
        bool isCurDup = false;
        ListNode* resHead = NULL;
        ListNode* p = pHead;
        ListNode* pNext = p->next;
        while (p){
            // 遇到重复结点跳过，pNext 是下一个未判断是否重复的结点（1->1->2->2->3：p = 1, pNext = 2）
            while (pNext && pNext->val == p->val){
                isCurDup = true;
                pNext = pNext->next;
            }
            // 将重复结点加入 set 中
            if(isCurDup){
                isDupSet.insert(p->val);
                isCurDup = false;
            }
            p = pNext;
            if(p){
                pNext = p->next;
            }
        }
        p = pHead;
        while (p){
            p = FindNextNode(p, isDupSet);  // 找到 p
            if(p){
                pNext = FindNextNode(p->next, isDupSet);    // 找到 pNext

                // 设置头节点
                if(resHead == NULL){
                    resHead = p;
                }
                // 设置 p 指向 pNext
                p->next = pNext;
                p = p->next;
            }
        }
        return resHead;
    }
    ListNode* FindNextNode(ListNode* pStart, set<int> isDupSet){
        while (pStart && isDupSet.find(pStart->val) != isDupSet.end()){
            pStart = pStart->next;
        }
        return pStart;
    }
    /*
     * 删除链表中重复的结点
     *
     * 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
     * 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
     *
     * 思路：给链表加个头节点，遍历链表
     */
    ListNode* deleteDuplicateNode2(ListNode* pHead){
        if(pHead == NULL)
            return NULL;
        if(pHead->next == NULL)     // 只有一个节点
            return pHead;
        ListNode* preHead = new ListNode(0);    // 添加头节点
        preHead->next = pHead;
        ListNode* curPre = preHead;
        ListNode* cur = pHead;
        ListNode* curNext = cur->next;
        bool isCurDup = false;
        while (cur){
            while (curNext && curNext->val == cur->val){
                isCurDup = true;
                curNext = curNext->next;
            }
//            if(cur == pHead && curNext == NULL) // 所有结点值都重复的情况
//                return NULL;
            if(!isCurDup){  // 当前节点不是重复结点时，将当前结点链接在结果链表中
                curPre->next = cur;
                curPre = cur;
                cur = cur->next;
                if(cur){
                    curNext = cur->next;
                }
            }else{  // 当前结点是重复结点时， 跳过
                if(curNext == NULL)     // 后面没有结点，并且当前结点是重复结点时，curPre 指向空
                    curPre->next = NULL;

                cur = curNext;
                if(cur){
                    curNext = cur->next;
                }
                isCurDup = false;
            }
        }
        return preHead->next;
    }

    /*
     * 打印链表
     */
    void PrintList(ListNode* pHead){
        ListNode* p = pHead;
        while (p){
            if(p->next)
                cout<<p->val<<"->";
            else
                cout<<p->val<<endl;
            p = p->next;
        }
    }
    /*
    * 使用数组创建单链表
    */
    ListNode* createList(vector<int> array){
        if(array.size() <= 0)
            return NULL;
        ListNode* pHead = new ListNode(array[0]);
        ListNode* pPreTemp = pHead;
        for(int i=1; i<array.size(); i++){
            ListNode* pTemp = new ListNode(array[i]);
            pPreTemp->next = pTemp;
            pPreTemp = pTemp;
        }
        return pHead;
    }
    /*
     * 把数组排成最小的数
     *
     * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
     *
     * 思路：首先将数组转化为 string 类型
 *           自定义比较器， 使用 sort 对数组进行排序
*           （比较函数必须写在类外部（全局区域）或声明为静态函数）
     */
    string PrintMinNumber(vector<int> arr){
        // 转换成 string 处理
        vector<string> seq;
        for(auto it : arr){
            seq.push_back(to_string(it));
        }
        sort(seq.begin(), seq.end(), comparator);
        string res;
        for(auto it : seq){
            res += it;
        }
        return res;
    }
    // 比较器：ab < ba 返回 true
    bool static comparator(const string &a, const string &b){
        return (a+b).compare(b+a) < 0 ? true : false;
    }
    /*
     * 丑数
     *
     * 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
     * 例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第 k 个丑数。
     *
     * 思路1：使用一个 set 存储当前出现的丑数 （不过，但测试用例正确）
     */
    int GetUglyNumber(int k){
        if(k <= 0){
            return 0;
        }
        set<int> UglySet;
        UglySet.insert(1);
        UglySet.insert(2);
        UglySet.insert(3);
        UglySet.insert(4);
        UglySet.insert(5);
        if(k <= UglySet.size()){
            set<int>::iterator it = UglySet.begin();
            for(int i=0; i<k-1; i++){
                ++it;
            }
            return *it;
        }
        int i = 6;
        while (UglySet.size() < k){
            if((i % 2 == 0) && (UglySet.find(i / 2) != UglySet.end()) || (i % 3 == 0) && (UglySet.find(i / 3) != UglySet.end()) || (i % 5 == 0) && (UglySet.find(i / 5) != UglySet.end())){
                UglySet.insert(i);
            }
            i++;
        }
        return *(--UglySet.end());
    }

    /*
     * 丑数
     *
     * 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
     * 例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第 k 个丑数。
     *
     * 思路2：逐个判断是否是丑叔，思路简单，但是计算冗余，因为越到后面很多都不是丑数也在计算。（太慢！不过）
     */
    int GetUglyNumber2(int k){
        if(k <= 0)
            return 0;
        if(k == 1)
            return 1;
        int count = 1;
        int i = 2;
        while (count != k){
            if(isUglyNumber(i)){
                count++;
            }
            i++;
        }
        return i - 1;
    }
    bool isUglyNumber(int n){
        while (n % 2 == 0){
            n /= 2;
        }
        while (n % 3 == 0){
            n /= 3;
        }
        while (n % 5 == 0){
            n /= 5;
        }
        return n == 1 ? true : false;
    }
    /*
     * 丑数
     *
     * 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
     * 例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第 k 个丑数。
     *
     * 思路3：第一个丑数是 1 ， 第二个丑数是 1*2, 1*3, 1*5 中的最小值，
     * 使用p1, p2, p3分别代表丑数列表中乘以系数 2、3、5 的下标， 一开始p1, p2, p3都指向第一个丑数
     * uglyArr[p1] * 2 、uglyArr[p2] * 3、uglyArr[p3] * 5 是下一个丑数的候选集，在其中找出最小值加入丑数列表中
     * 并且更新选择加入丑数列表的丑数的下标，
     * 即若选择将uglyArr[p1] * 2加入丑数列表，则更新 p1
     */
    int GetUglyNumber3(int k){
        if(k <= 0)
            return 0;
        if(k == 1)
            return 1;
        vector<int> uglyArr;
        uglyArr.push_back(1);
        int p1 = 0, p2 = 0, p3 = 0;
        while (uglyArr.size() < k){
            int v1 = uglyArr[p1] * 2;
            int v2 = uglyArr[p2] * 3;
            int v3 = uglyArr[p3] * 5;
            int nextUgly = std::min(v1, std::min(v2, v3));
            if(nextUgly == v1){
                p1++;
            } else if(nextUgly == v2){
                p2++;
            } else{
                p3++;
            }
            if(nextUgly != uglyArr[uglyArr.size() - 1])
                uglyArr.push_back(nextUgly);
        }
        return uglyArr[uglyArr.size() - 1];
    }
    /*
     * 第一个只出现一次的字符
     *
     * 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,
     * 并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
     *
     * 思路：使用一个数组做哈希表，下标是key， 数组中存的值是value代表每个字符出现几次
     *      遍历一遍字符串，更新哈希表
     *      再遍历一遍字符串，找到第一个在哈希表中value为1的字符
     */
    int FirstNotRepeatChar(string str){
        if(str.length() == 0)
            return -1;
        if(str.length() == 1)
            return 0;
        int hashmap[256]{0};   // char 型1字节有256种字符，全部初始化为0
        for(char* pCh = &str[0]; *pCh != '\0'; ++pCh){
            hashmap[*pCh]++;
        }
        int index = 0;
        for(char* pCh = &str[0]; *pCh != '\0'; ++pCh){
            if(hashmap[*pCh] == 1)
                return index;
            index++;
        }
        return -1;
    }
    /*
     * 两个链表的第一个公共结点
     *
     * 输入两个链表，找出它们的第一个公共结点。
     *
     * 思路1：遍历list1 记录长度 len1， 遍历 list2 记录长度 len2
     *        设置两个指针分别指向两个链表头，指向长度大的链表的指针先走|len1 - len2|步
     *        然后两个指针一起走，直到遇到相同结点或者某个指针走到链表尾停止
     */
    ListNode* FindCommonNode(ListNode* pHead1, ListNode* pHead2){
        if(pHead1 == NULL || pHead2 == NULL)
            return NULL;
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        int len1 = 0;
        int len2 = 0;
        while (p1 || p2){
            if(p1){
                len1++;
                p1 = p1->next;
            }
            if(p2){
                len2++;
                p2 = p2->next;
            }
        }
        // p1 指向长度长的指针头， p2 指向长度短的指针头
        if(len1 - len2 > 0){
            p1 = pHead1;
            p2 = pHead2;
        } else{
            p1 = pHead2;
            p2 = pHead1;
        }
        int temp = abs(len1 - len2);
        while (temp > 0){
            p1 = p1->next;
            temp--;
        }
        while (p1 && p2){
            if(p1 == p2)
                return p1;
            p1 = p1->next;
            p2 = p2->next;
        }
        return NULL;
    }
    /*
     * 数字在排序数组中出现的次数
     *
     * 统计一个数字在排序数组中出现的次数。
     * 思路：先找到k的下标，找到第一个k，再统计k出现的次数（未通过）
     */
    int GetTimesOfK(vector<int> arr, int k){
        if(arr.size() == 0)
            return 0;
        int index = GetIndexOfK(arr, 0, arr.size() - 1, k);
        int count = 0;
        if(index != -1){
            // 找到第一个 k
            int i = index;
            while(i-1 >= 0 && arr[i-1] == k){
                i--;
            }
            while (arr[i] == k){
                count++;
                i++;
            }
        }
        return count;
    }
    int GetIndexOfK(vector<int> arr, int start, int end, int k){
        if(start <= end){
            int mid = (start + end) / 2;
            if(arr[mid] == k)
                return mid;
            else if(arr[mid] < k){
                return GetIndexOfK(arr, mid + 1, end, k);
            }else{
                return GetIndexOfK(arr, start, mid - 1, k);
            }
        }
        return -1;
    }
    /*
     * 数字在排序数组中出现的次数
     *
     * 统计一个数字在排序数组中出现的次数。
     * 思路：先找到第一个k的下标，再找到k后面第一个数的下标，返回两个下标的差（通过）
     */
    int GetTimesOfK2(vector<int> arr, int k){
        if(arr.size() == 0)
            return 0;
        int kStartIndex = GetIndexOfK(arr, k);
        int kNextStartIndex = GetIndexOfK(arr, k+1);
        return kNextStartIndex - kStartIndex;
    }
    // 得到整数 k 在有序数组中的下标，没有则返回第一个大于它的整数的下标
    int GetIndexOfK(vector<int> arr, int k){
        int start = 0, end = arr.size() - 1;
        while (start <= end){
            int mid = (start + end) / 2;
            if(arr[mid] >= k){
                end = mid;
            } else{
                start = mid + 1;
            }
        }
        if(arr[end] < k)
            return end + 1;
        return end;
    }

    /*
     * 数组中只出现一次的数字
     *
     * 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
     */
    void FindNumsAppearOnce(vector<int> arr, int* num1, int* num2){
        int temp = 0;
        for(int i=0; i<arr.size(); i++){
            temp = temp ^ arr[i];
        }
        int flag = 1;
        while ((temp & flag) == 0){
            flag = flag << 1;
        }
        vector<int> arr0;
        vector<int> arr1;
        for(int i=0; i<arr.size(); i++){
            if((arr[i] & flag) == 0){
                arr0.push_back(arr[i]);
            } else{
                arr1.push_back(arr[i]);
            }
        }
        *num1 = *num2 = 0;
        for(int i=0; i<arr0.size(); i++){
            *num1 = *num1 ^ arr0[i];
        }
        for(int i=0; i<arr1.size(); i++){
            *num2 = *num2 ^ arr1[i];
        }
    }
    /*
     * 旋转数组中找某个值的位置
     *
     * 思路：
     *      arr[mid] == k，返回 mid
     *      arr[mid] < arr[end] ，右边数组有序，若arr[mid] < k <= arr[end]，在右边数组查找，否则在左边数组查找
     *      arr[mid] >= arr[end]，左边数组有序，若arr[start] <= k < arr[mid]，在左边数组查找，否则在右边数组查找
     */
    int FindK(vector<int> arr, int k){
        int begin = 0, end = arr.size() - 1;
        return FindK(arr, begin, end, k);
    }
    int FindK(vector<int> arr, int start, int end, int k){
        if(start <= end){
            int mid = (start + end) / 2;
            if(arr[mid] == k)
                return mid;
            if(arr[mid] < arr[end]){    // 右边数组有序
                if(arr[mid] < k && k <= arr[end]){
                    return FindK(arr, mid+1, end, k);
                } else{
                    return FindK(arr, start, mid - 1, k);
                }
            } else{     // 左边数组有序
                if(arr[start] <= k && k < arr[mid]){
                    return FindK(arr, start, mid - 1, k);
                } else{
                    return FindK(arr, mid + 1, end, k);
                }
            }
        }
        return -1;
    }

    /*
     * 左旋转字符串
     *
     * 对于一个给定的字符序列 S，请你把其循环左移 n 位后的序列输出
     *
     * 例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”
     *
     * 思路：前 n 个逆置，n 后面的字符串也逆置，最后整体逆置
     *      如果 n > 字符串长度，n = n % len;
     */
    string LeftRotateString(string str, int n) {
        if(str.length() <= 1)
            return str;
        if(n > str.length()){
            n = n % str.length();
        }
        RotateString(str, 0, n-1);
        RotateString(str, n, str.length()-1);
        RotateString(str, 0, str.length()-1);
        return str;
    }
    void RotateString(string &str, int start, int end){
        while(start < end){
            swap(str[start++], str[end--]);
        }
    }

    /*
     * 翻转单词顺序列
     *
     * 例如，“student. a am I” 翻转成“I am a student.”
     *
     * 思路1：逆置每个单词，再整体逆置
     */
    string ReverseSentence(string str){
        if(str.length() == 0)
            return str;
        int wordStart = 0, wordEnd = 0;
        while (str[wordStart] != '\0'){
            while (str[wordStart] != '\0' && str[wordStart] == ' '){
                wordStart++;
            }
            wordEnd = wordStart;
            while (str[wordEnd] != '\0' && str[wordEnd] != ' '){
                wordEnd++;
            }
            wordEnd--;
            Reverse(str, wordStart, wordEnd);
            wordStart = wordEnd + 1;
        }
        Reverse(str, 0, str.length() - 1);
        return str;
    }
    void Reverse(string &str, int start, int end){
        while (start < end){
            swap(str[start++], str[end--]);
        }
    }
    /*
     * 扑克牌顺子
     *
     * 抽出5张牌 , 看抽出的牌能不能组成顺子，牌 1->13， 0 代表大小王
     *
     * 0可以代表任何数，检查数组能不能组成连续序列
     * 3 0 4 1 -> 1 2 3 4 (用 0 代表 2)
     *
     * 思路1：
     *      Step1：排序
     *
     *      Step2：计算0的个数
     *
     *      Step3：计算相邻数字的“距离”，并且保证除0外相邻数字不能重复。
     *
     *      Step4：比较“距离”是否小于0的个数。
     */
    bool IsContinuous(vector<int> arr){
        if(arr.size() == 0)
            return false;
        sort(arr.begin(), arr.end());
        int i=0;
        int numsOf0 = 0;
        int sum = 0;
        while (i < arr.size() && arr[i] == 0){
            numsOf0++;
            i++;
        }
        int j = i + 1;
        while (j < arr.size()){
            if(arr[j] == arr[i])
                return false;
            sum += arr[j++] - arr[i++] - 1;
        }
        if(numsOf0 >= sum){
            return true;
        }
        return false;
    }
    /*
     * 扑克牌顺子
     *
     * 抽出5张牌 , 看抽出的牌能不能组成顺子，牌 1->13， 0 代表大小王
     *
     * 0可以代表任何数，检查数组能不能组成连续序列
     * 3 0 4 1 -> 1 2 3 4 (用 0 代表 2)
     *
     * 思路2：
     *          遍历数组：
     *          1 使用一个数组记录有没有重复数字，遍历时判断，有重复数字返回false
     *          2 max 和 min 记录最大值最小值
     *          3 max - min < 4 代表可以组成顺子
     */
    bool IsContinuous2(vector<int> arr){
        if(arr.size() != 5)
            return false;
        int dup[14]{0};     // 扑克牌：1->13 数组下标代表牌号
        int max = 0, min = 14;
        for(int i=0; i<arr.size(); i++){
            if(arr[i] == 0)     // 遇到 0 时跳过，避免把 0 赋值给 min
                continue;
            dup[arr[i]]++;
            if(dup[arr[i]] > 1){
                return false;
            }
            if(arr[i] > max){
                max = arr[i];
            }
            if(arr[i] < min){
                min = arr[i];
            }
        }
        if(max - min >= 0 && max - min < 5){
            return true;
        }
        return false;
    }
    /*
     * 孩子们的游戏(圆圈中最后剩下的数)
     *
     *
     */
    int LastRemaining_Solution(int n, int m){

    }
    /*
     * 和为S的连续正数序列
     *
     * 找出所有和为S的连续正数序列(至少包括两个数)
     *
     * 例如：18,19,20,21,22 的和是 100
     *       9,10,11,12,13,14,15,16 的和是 100
     */
    vector<vector<int> > FindContinuousSequence(int sum){

    }
    /*
     * 数组中的逆序对
     *
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
     * 输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。
     * 即输出P%1000000007
     *
     * 题目保证输入的数组中没有的相同的数字
     * 数据范围：
     * 对于%50的数据,size<=10^4
     *
     * 对于%75的数据,size<=10^5
     *
     * 对于%100的数据,size<=2*10^5
     *
     * 示例：
     * 输入：1,2,3,4,5,6,7,0
     * 输出：7
     */
    int InversePairs(vector<int> arr){

    }
    /*
     * 整数中1出现的次数（从1到n整数中1出现的次数）
     *
     * 例如：算出100~1300的整数中1出现的次数
     *
     * 思路：
     *      对于数字n，计算它的第i(i从1开始，从右边开始计数)位数上包含的数字1的个数：
     *
     *      假设第i位上的数字为x的话，则
     *
     *      1.如果x > 1的话，则第i位数上包含的1的数目为：(高位数字 + 1）* 10 ^ (i-1)  (其中高位数字是从i+1位一直到最高位数构成的数字)
     *
     *      2.如果x < 1的话，则第i位数上包含的1的数目为：(高位数字 ）* 10 ^ (i-1)
     *
     *      3.如果x == 1的话，则第i位数上包含1的数目为：(高位数字) * 10 ^ (i-1) +(低位数字+1)   (其中低位数字时从第i - 1位数一直到第1位数构成的数字)
     */
    int NumberOf1Between1AndN_Solution(int n) {


    }





private:
    stack<int> stack1;
    stack<int> stack2;

    stack<int> dataStack;
    stack<int> minStack;

};
int main() {
    Solution solution;

    vector<int> arr = {1,3,0,5,0};
    cout<<solution.IsContinuous2(arr);

//    string str = "   i am a student.   ";
//    string res = solution.ReverseSentence(str);
//    cout<<res;

//    string str = "";
//    string res = solution.LeftRotateString(str, 5);
//    cout << res;

//    vector<int> arr1 = {5, 6, 7, 8, 0, 2, 3};
//    cout<< solution.FindK(arr1, 6);



//    vector<int> arr = {2,4,3,6,3,2,5,5};
//    int* num1;
//    int* num2;
//    solution.FindNumsAppearOnce(arr, num1, num2);
//    cout<<*num1<<endl<<*num2;

//    vector<int> arr = {2,2};
//    cout<<solution.GetTimesOfK2(arr, 3);

//    vector<int> arr1 = {1,2,3,3,4,4,5};
//    vector<int> arr2 = {1,2,3,3,4,4,5};
//    ListNode* pHead1 = solution.createList(arr1);
//    ListNode* pHead2 = solution.createList(arr2);
//    ListNode* p = pHead1;
//    for(int i=0; i<2; i++){
//        p = p->next;
//    }
//    p->next = pHead2->next;
//    ListNode* res = solution.FindCommonNode(pHead1, pHead2);
//    solution.PrintList(res);


//    string str = "aab";
//    cout<<solution.FirstNotRepeatChar(str);

//    cout<<solution.GetUglyNumber3(9);

//    int arr[5];
//    for(auto it: arr){  // 脏值 2004489063 -1 36 12 7998544
//        cout<<it<<" ";
//    }
//    cout<<endl;
//    int arr1[5] = {1};   // c++11中，中间的赋值号可以省略，即可写成int arr1[5]{}; 与 int arr1[5]{0};等价
//    for(auto it: arr1){ // 0 0 0 0 0
//        cout<<it<<" ";
//    }

//    vector<int> arr = {3, 22, 321};
//    string res = solution.PrintMinNumber(arr);
//    cout<<res<<endl;

//    vector<int> arr = {1,2,3,3,4,4,5};
//    ListNode* pHead = solution.createList(arr);
//    solution.PrintList(pHead);
//    ListNode* res = solution.deleteDuplicateNode2(pHead);
//    solution.PrintList(res);

//    vector<int> preOrd = {4,2,1,3,7,5,6,8};
//    vector<int> inOrd = {1,2,3,4,5,6,7,8};
//    TreeNode* pRoot = solution.reConstructBinaryTree(preOrd, inOrd);
//    cout<<"TreeHeight: "<<solution.BTHeight(pRoot)<<endl;
//    vector<vector<int> > res = solution.PrintBTByZhi(pRoot);
//    for(int i=0; i<res.size(); i++){
//        for(int j=0; j<res[i].size(); j++){
//            cout<<res[i][j]<<" ";
//        }
//        cout<<endl;
//    }


//    vector<int> res = solution.PrintFromTopToBottom(pRoot);
//    for(auto it: res){
//        cout<<it<<" ";
//    }
//    TreeNode* KthNode = solution.KthNode(pRoot, 9);
//    if(KthNode){
//        cout<<endl<<KthNode->val;
//    }



//    vector<int> arr = {6,-3,-2,7,-15,1,2,2};
//    cout<<solution.FindGreatestSumOfSubArray(arr);

//    vector<int> arr = {4,5,1,6,2,7,3,8};
//    vector<int> res = solution.minHeapSort(arr);
//    for(auto it : res){
//        cout<<it<<" ";
//    }

//    vector<int> arr = {4,5,1,6,2,7,3,8};
//    vector<int> res = solution.maxHeapSort(arr);
//    for(auto it : res){
//        cout<<it<<" ";
//    }

//    vector<int> arr = {4,5,1,6,2,7,3,8};
//    vector<int> res = solution.GetLeastNumbers(arr, 3);
//    for(auto it: res){
//        cout<<it<<" ";
//    }

//    vector<int> arr = {2, 3, 2, 3, 2, 3, 2};
//    cout<<solution.MoreThanHalfNum(arr);

//    string str = "aa";
//    vector<string> res = solution.Permutation(str);
//    for(auto it : res){
//        cout<<it<<endl;
//    }

//    vector<int> sequence = {5,7,4,9,11,10,8};
//    cout<<solution.VerifySquenceOfBST(sequence)<<endl;

//    int n;
//    while (cin>>n){
//        int val;
//        vector<int> pushV(n);
//        vector<int> popV(n);
//        for (int i = 0; i < n; ++i) {
//            cin>>val;
//            pushV[i] = val;
//        }
//        for (int i = 0; i < n; ++i) {
//            cin>>val;
//            popV[i] = val;
//        }
//        if(solution.isPopOrder(pushV, popV)){
//            cout<<"1"<<endl;
//        }else{
//            cout<<"0"<<endl;
//        }
//    }


//    vector<vector<int >> matrix(3, vector<int>(4));
//    matrix = {{1}, {2}, {3}, {4}, {5}};
//    vector<int> res = solution.printMatrix(matrix);
//    for(auto it: res){
//        cout<<it<<" ";
//    }

//    double base;
//    int exponent;
//    while (cin>>base>>exponent){
//        cout<<solution.Power(base, exponent)<<endl;
//    }
    return 0;
}