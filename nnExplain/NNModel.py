import numpy as np
from enum import Enum
from collections import namedtuple
from namedlist import namedlist
# import z3
import tensorflow as tf
import copy
import time
from typing import List
import gurobipy
from . import utils as U

ExprOp = Enum('ExprOp', 'eq ge le gt lt')

def op_to_symbol(op:ExprOp)->str:
    if op == ExprOp.gt:
        return '>'
    if op == ExprOp.lt:
        return '<'
    if op == ExprOp.eq:
        return '='
    if op == ExprOp.ge:
        return '>='
    if op == ExprOp.le:
        return '<='
    raise ValueError("Unregonized operator: "+str(op))

"""
Generate a z3 expression
"""
def apply_expr_op_z3(lh, op, rh):
    if op == ExprOp.gt:
        return lh > rh
    if op == ExprOp.lt:
        return lh < rh
    if op == ExprOp.eq:
        return lh == rh
    if op == ExprOp.ge:
        return lh >= rh
    if op == ExprOp.le:
        return lh <= rh
    raise ValueError("Unregonized operator: "+str(op))

LayerKind = Enum('LayerKind', 'input dense relu dropout conv pool reshape diff_input absolute')


"""
Store the information of one layer.
kind: 'input' 'dense' 'relu' 'dropout'
var: symbolic tensor of that layer
weights: only applicable to 'dense'. A concrete tensor that stores the weights
biases: only applicable to 'dense'. A concrete tensor that stores the biases
When the layer is a maxpool layer, we use weights to store ksize respectively
strides: only used for convolutional and maxpool layers
padding_type: only used for convulitional and maxpool layers. Options: SAME | VALID. None means VALID.
"""
Layer = namedlist('Layer', 'kind var weights biases strides padding_type', default = None)

"""
idx: the index of the relu layer among all layers
record: a list of 1's and 0's which represent whether a given relu is activated
"""
ReluRecord = namedtuple('ReluRecord', 'idx record')

'''
A set of linear expressions which are in the form of XA+B.
'''
class LinearExpressions:
    def __init__(self,A,B):
        self.A = np.array(A)
        self.B = np.array(B)
    
    '''
    Add a single linear expression to the set
    '''
    def extend(self, a, b):
        toAppend = np.array([a])
        nA = np.concatenate((self.A, toAppend.T), axis = 1)
        nB = np.append(self.B,b)
        return LinearExpressions(nA, nB)

    def extend1(self, other):
        nA = np.concatenate((self.A, other.A), axis = 1)
        nB = np.concatenate((self.B, other.B))
        return LinearExpressions(nA, nB)

    def __eq__(self, o: object) -> bool:
        if isinstance(self, o.__class__):
            return np.array_equal(self.A, o.A) and np.array_equal(self.B, o.B)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

'''
Represent the linear constraints where XA + B <ops> 0
'''
class LinearConstraintSet(LinearExpressions):
    def __init__(self, A, B, ops, internal_edges = None):
        super().__init__(A,B)
        self.ops = ops
        self.UBs = None
        self.LBs = None
        if internal_edges is None:
            self.internal_edges = [False] * len(self.ops)
        else:
            self.internal_edges = internal_edges

        # Cache fields for speeding up the computation
        self.sur_chk_vars = None
        self.sur_chk_model = None
        self.sur_chk_cons = None

    def extend(self, other):
        if len(self.A) == 0:
            nA = np.copy(other.A)
        else:
            nA = np.concatenate((self.A, other.A), axis = 1)
        nB = np.concatenate((self.B, other.B))
        nops = np.concatenate((self.ops, other.ops))
        ninternal_edges = self.internal_edges + other.internal_edges
        ret = LinearConstraintSet(nA, nB, nops, ninternal_edges)
        return ret

    """
    Solve the linear constraints. 
    @returns: an array representing the assignment, or None for UNSAT
    """
    def solve(self):
        start_time = time.time()
        result = self.solve_gurobi()
        end_time = time.time()
        U.verbose('Gurobi solving time: '+str(end_time-start_time), 2)

        # start_time = time.time()
        # result = self.solve_z3()
        # end_time = time.time()
        # print('Z3 solving time: '+str(end_time - start_time))
        return result

    def minimize(self, weights: List[float]):
        m,vars,cons = self.constructGurobiModel()

        # add objective
        obj = 0
        for (w,var) in zip(weights, vars):
            obj += w*var
        m.setObjective(obj, gurobipy.GRB.MINIMIZE)

        start_time = time.time()
        m.optimize()
        end_time = time.time()
        U.verbose('Raw Gurobi solving time: ' + str(end_time - start_time), 2)

        if m.status == gurobipy.GRB.INFEASIBLE:
            if U.VERBOSE_LEVEL >= 3:
                m.write('debug.lp')
                m.computeIIS()
                m.write("./debug.ilp")
            return None

        ret = []

        for v in range(len(vars)):
            ret.append(vars[v].x)

        return ret

    def constructGurobiModel(self):

        m = gurobipy.Model('Linear Region Feasibility')
        if U.VERBOSE_LEVEL < 3:
            m.Params.OutputFlag = 0
        num_vars = len(self.A)
        # m.Params.FeasibilityTol=1e-9
        m.Params.DualReductions = 0
        m.Params.LogFile = ''
        vars = m.addVars(list(range(num_vars)), lb = -gurobipy.GRB.INFINITY, vtype = gurobipy.GRB.CONTINUOUS)

        m.update()

        constrs = []

        t1 = time.time()

        for (colidx, (a, b, op)) in enumerate(zip(self.A.T, self.B, self.ops)):
            expr = 0
            for (rowIdx, coe) in enumerate(a):
                expr += coe * vars[rowIdx]
            expr += b

            if op == ExprOp.ge:
                cons = m.addConstr(expr, gurobipy.GRB.GREATER_EQUAL, 0)
            elif op == ExprOp.le:
                cons = m.addConstr(expr, gurobipy.GRB.LESS_EQUAL, 0)
            elif op == ExprOp.eq:
                cons = m.addConstr(expr, gurobipy.GRB.EQUAL, 0)
            elif op == ExprOp.gt:
                cons = m.addConstr(expr, gurobipy.GRB.GREATER_EQUAL, U.LINEAR_TOLERANCE)
            elif op == ExprOp.lt:
                cons = m.addConstr(expr, gurobipy.GRB.LESS_EQUAL, -U.LINEAR_TOLERANCE)
            else:
                raise ValueError('Unknown operator type '+str(op))
            constrs.append(cons)

        return (m, vars, constrs)


    def solve_gurobi(self):
        start_time = time.time()
        m,vars,cons = self.constructGurobiModel()
        U.verbose('model construction time: '+str(time.time() - start_time), 2)
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        U.verbose('Raw Gurobi solving time: '+str(end_time-start_time), 2)

        if m.status == gurobipy.GRB.INFEASIBLE:
            # m.write('debug.lp')
            # m.computeIIS() # got error sometimes, might be numerical issue
            # m.write("./debug.ilp")
            return None

        ret = []

        for v in range(len(vars)):
            ret.append(vars[v].x)

        U.verbose('Result collection time: '+str(time.time()-end_time), 2)

        return ret

    def solve_z3(self):
        z3Solver  = z3.Solver()
        z3.set_param(verbose = 10)
        ashape = self.A.shape
        bshape = self.B.shape
        assert(len(ashape) == 2 and len(bshape) == 1)
        assert(ashape[1] == bshape[0])
        # Create variables
        vars : List[z3.Real] = []
        for i in range(0, ashape[0]):
            vars.append(z3.Real(str(i)))

        for (colidx, (a,b, op)) in enumerate(zip(self.A.T, self.B, self.ops)):
            expr = 0
            for (rowIdx, coe) in enumerate(a):
                expr += coe * vars[rowIdx]
            expr += b
            z3Solver.add(apply_expr_op_z3(expr, op, 0))
        start_time = time.time()
        cresult = z3Solver.check()
        end_time = time.time()
        U.verbose('Raw z3 solving time: '+str(end_time-start_time), 2)
        if cresult != z3.sat:
            return None
        m = z3Solver.model()
        ret = []
        for i in range(0, len(m)):
            var = m[i]
            v = float(m[var].numerator_as_long())/float(m[var].denominator_as_long())
            ret.append(v)
        return ret

    """
    Shift the region by dist_array
    """
    def shift(self, dist_array):
        ret_A = np.copy(self.A)
        ret_B = self.B + np.dot(dist_array, ret_A)
        ret_ops = copy.copy(self.ops)
        ret = LinearConstraintSet(ret_A, ret_B, ret_ops, copy.copy(self.internal_edges))
        return ret

    """
    Generate a new expression with variables fixed to given constants
    """
    def fix_vars(self, var_list: List[int], cons_list: List[int]):
        a_shape = self.A.shape
        nAt = np.empty((a_shape[1], a_shape[0] - len(var_list)))
        nB = []
        for (ridx, (row, b)) in enumerate(zip(self.A.T, self.B)):
            nb = b
            for (idx, v) in zip(var_list, cons_list):
                nb += row[idx] * v

            var_list_set = set(var_list)

            nrow = []
            for i,x in enumerate(row):
                if i not in var_list_set:
                    nrow.append(x)
            # nrow = [x for i,x in enumerate(row) if i not in var_list ]
            nAt[ridx] = nrow
            nB.append(nb)
        nA = nAt.T
        return LinearConstraintSet(nA, nB, self.ops, copy.copy(self.internal_edges))

    """
    Check if a given X satisfies the current linear constraints
    """
    def eval_gurobi(self, X):
        assert(len(self.A) == len(X))
        cons = self.fix_vars(list(range(len(self.A))), X)
        return cons.solve() is not None

    def eval(self, X):
        assert(len(self.A) == len(X))
        a_shape = self.A.shape
        for ((row, b), op) in zip(zip(self.A.T, self.B), self.ops):
            nb = b
            for (idx, v) in enumerate(X):
                nb += row[idx] * v

            if op == ExprOp.le and nb > 0:
                return False
            if op == ExprOp.lt and nb >= 0:
                return False
            if op == ExprOp.eq and nb !=0:
                return False
            if op == ExprOp.ge and nb < 0:
                return False
            if op == ExprOp.gt and nb <= 0:
                return False

        return True


    """
    Fix and remove certain variables. Generate a new expression
    """
    def fix_and_remove_vars(self, rm_list: List[int], cons_list: List[int]):
        var_indices = list(range(len(self.A)))
        kept_indices = [x for x in var_indices if x not in rm_list]
        nA = self.A[kept_indices]
        rm_A = self.A[rm_list]
        nB = self.B + np.dot(cons_list, self.A[rm_list])
        return LinearConstraintSet(nA,nB, self.ops, copy.copy(self.internal_edges))

    def get_bounds(self):
        if self.UBs is None or self.LBs is None:
            m,vars,cons = self.constructGurobiModel()
            ret = self.get_UB_LB(m,vars)
            if ret is not None:
                self.UBs, self.LBs = ret
            return ret
        return self.UBs, self.LBs

    def get_UB_LB(self, m, vars):
        LBs = []
        UBs = []
        for k,v in vars.iteritems():
            m.setObjective(v, gurobipy.GRB.MINIMIZE)
            m.update()
            m.optimize()
            if m.status == gurobipy.GRB.INFEASIBLE:
                m.write('debug.lp')
                m.computeIIS() # got error sometimes, might be numerical issue
                m.write("./debug.ilp")
                return None
            if m.status != gurobipy.GRB.OPTIMAL:
                LBs.append(-U.LINEAR_TOLERANCE_SCALE)
            else:
                LBs.append(m.getObjective().getValue())
            m.setObjective(v, gurobipy.GRB.MAXIMIZE)
            m.update()
            m.optimize()
            if m.status != gurobipy.GRB.OPTIMAL:
                UBs.append(U.LINEAR_TOLERANCE_SCALE)
            else:
                UBs.append(m.getObjective().getValue())
        return (UBs, LBs)

    """
    Remove redundant by using UB/LB method.
    The idea is that we can get the UB and LB of each variable and add these bounds as additional constraints.
    A constraint is redundant if it is trivially implied by these bounds constraints.
    This method should work well when the number of constraints is large and the number of vars is small.
    """
    def removeRedundantBounds(self):
        m,vars,cons = self.constructGurobiModel()
        UBs,LBs = self.get_UB_LB(m,vars)

        rAT = []
        rB = []
        rops = []
        ries = []
        for a, b, op, rie in zip(self.A.T, self.B, self.ops, self.internal_edges):
            if op == ExprOp.eq:
                continue
            sense = (op == ExprOp.ge)
            lhs = b
            for i,v in enumerate(a):
                if (v >= 0 and sense) or (v < 0 and not sense):
                    lhs += LBs[i] * v
                else:
                    lhs += UBs[i] * v
            if (sense and lhs <= 0) or (not sense and lhs >= 0):
                rAT.append(a)
                rB.append(b)
                rops.append(op)
                ries.append(rie)
        # add bounds
        for idx, (lb, ub, var) in enumerate(zip(LBs,UBs, vars)):
            if lb != -gurobipy.GRB.INFINITY:
                a = np.zeros(len(vars))
                a[idx] = 1
                rAT.append(a)
                rB.append(-lb)
                rops.append(ExprOp.ge)
                ries.append(False)
            if ub != gurobipy.GRB.INFINITY:
                a = np.zeros(len(vars))
                a[idx] = 1
                rAT.append(a)
                rB.append(-ub)
                rops.append(ExprOp.le)
                ries.append(False)

        return LinearConstraintSet(np.array(rAT).T, rB, rops, ries)

    """
    Return a set of linear constraints with redundant constraints removed
    """
    def removeRedundant(self):
        ret = self.copy()
        U.verbose('Initial number of constraints: '+str(len(ret.ops)), 2)
        U.verbose('Step 1: remove trivially held constraints', 2)
        trivial_constrs = []
        for (idx, (a,b, op)) in enumerate(zip(ret.A.T, ret.B, ret.ops)):
            if not a.any():
                trivial_constrs.append(idx)
        ret.A = np.delete(ret.A, trivial_constrs, 1)
        ret.B = np.delete(ret.B, trivial_constrs, 0)
        ret.ops = np.delete(ret.ops, trivial_constrs, 0)
        ret.internal_edges = np.delete(ret.internal_edges, trivial_constrs, 0)

        U.verbose('Current number of constraints: '+str(len(ret.ops)), 2)
        U.verbose('Step 2: remove constraints using the LB/UB method', 2)
        ret = ret.removeRedundantBounds()

        num_cons_before = len(ret.ops)
        U.verbose('Current number of constraints: '+str(num_cons_before), 2)
        U.verbose('Step 3: remove constraints implied by others using ilp', 2)

        while True:
            num_cons = len(ret.ops)
            for i in range(num_cons):
                i = num_cons - 1 - i # delete constraints in a reverse manner so bound constraitns would be got rid of first
                cop = ret.ops[i]
                redundant = False
                if cop == ExprOp.ge:
                    ret.ops[i] = ExprOp.lt
                elif cop == ExprOp.le:
                    ret.ops[i] = ExprOp.gt

                if cop != ExprOp.eq:
                    redundant = (ret.solve() is None)
                else:
                    redundant = True
                    ret.ops[i] = ExprOp.lt
                    redundant &= (ret.solve() is None)
                    if redundant:
                        ret.ops[i] = ExprOp.gt
                        redundant &= (ret.solve() is None)
                if redundant:
                    ret.A = np.delete(ret.A, i, 1)
                    ret.B = np.delete(ret.B, i, 0)
                    ret.ops = np.delete(ret.ops,i, 0)
                    ret.internal_edges = np.delete(ret.internal_edges, i, 0)
                    U.verbose('Remove {} out of {} constraints.'.format(num_cons_before - len(ret.ops), num_cons_before), 2)
                    break
                else:
                    ret.ops[i] = cop
            if len(ret.ops) == num_cons:
                break
        U.verbose('Current number of constraints: '+str(len(ret.ops)), 2)

        return ret

    def __eq__(self, o: object) -> bool:
        return self.__dict__ == o.__dict__

    def to_str(self, var_names):
        ret = ''
        for a, b,o in zip(self.A.T, self.B, self.ops):
            for (i,v) in enumerate(a):
                ret += (str(v) + '*'+var_names[i])
                ret += ' + '
            ret += (str(b)+' ' + op_to_symbol(o)+' 0\n')
        return ret

    """
    A special function that denormalize the linear region using mean and standard deviation.
    A' = A/std (column direction)
    B' = B - mean*A'
    """
    def denormalize(self, A_mean, A_std):
        A1 = self.A / np.array(A_std)[:,None]
        B1 = self.B - np.dot(A_mean, A1)
        return LinearConstraintSet(A1, B1, self.ops)

    def getNumDimensions(self):
        return len(self.A)

    def getVertices(self):
        ret = None
        if len(self.A) == 1:
            ret = self.getVertices_1D()
        if len(self.A) == 2:
            ret = self.getVertices_2D()
        if ret is None:
            raise ValueError("Cannot handle high dimenson yet.")
        return np.round(ret, decimals = U.STITCH_TOLERANCE_DIGITS).tolist()

    def getVertices_1D(self):
        ret = []
        work_copy = self.copy()
        n_cos = len(work_copy.ops)
        
        lb = -gurobipy.GRB.INFINITY
        ub = gurobipy.GRB.INFINITY

        for i in range(n_cos):
            a = self.A[0][i]
            b = self.B[i]
            if a == 0:
                continue
            v = (-b)/a
            op = work_copy.ops[i]
            if op == ExprOp.eq:
                print("Warning! Equality in the constraints.")
                lb = v
                ub = v
                break
            
            if op == ExprOp.ge or op == ExprOp.gt:
                if a > 0:
                    if lb < v:
                        lb = v
                else:
                    if ub > v:
                        ub = v
            
            if op == ExprOp.le or op == ExprOp.lt:
                if a > 0:
                    if ub > v:
                        ub = v
                else:
                    if lb < v:
                        lb = v

        ret = [lb, ub]

        if len(ret) != 2:
            print("Debug")
        assert(len(ret) == 2)

        return ret

    def get_suspicous_points(self, vertices):
        suspicous_vertices = []

        for (idx, v) in enumerate(vertices):
            for (idx1, v1) in enumerate(vertices):
                if idx >= idx1:
                    continue
                if abs(v1[0] - v[0]) <= U.LINEAR_TOLERANCE and abs(v1[1] - v[1]) <= U.LINEAR_TOLERANCE:
                    if v not in suspicous_vertices:
                        suspicous_vertices.append(v)
                    if v1 not in suspicous_vertices:
                        suspicous_vertices.append(v1)

        return suspicous_vertices


    """
    TODO: now only handles the 2D case, need to extend to higher dimensions
    Get the vertices of the linear region
    """
    def getVertices_2D(self):
        work_copy = self.copy()
        n_cos = len(work_copy.ops)
        edge_to_vertices = {}
        vertex_to_edges = {}
        for i in range(n_cos):
            for j in range(n_cos):
                if j > i:
                    op1 = work_copy.ops[i]
                    op2 = work_copy.ops[j]
                    work_copy.ops[i] = ExprOp.eq
                    work_copy.ops[j] = ExprOp.eq
                    vertex = work_copy.solve()
                    if vertex is not None:
                        vertex1 = vertex
                        vertex = tuple(vertex)
                        while vertex in vertex_to_edges:
                            # This is a bit hacky, but we want a different key in the map
                            vertex1[0] = vertex1[0] + U.LINEAR_TOLERANCE*0.1
                            vertex1[1] = vertex1[1] + U.LINEAR_TOLERANCE*0.1
                            vertex = tuple(vertex1)
                        if i not in edge_to_vertices:
                            edge_to_vertices[i] = []
                        if j not in edge_to_vertices:
                            edge_to_vertices[j] = []
                        edge_to_vertices[i].append(vertex)
                        edge_to_vertices[j].append(vertex)
                        vertex_to_edges[vertex] = [i,j]
                    work_copy.ops[i] = op1
                    work_copy.ops[j] = op2

        # print(vertex_to_edges)
        # print(edge_to_vertices)

        # Step 2: remove edges formed with suspicous points only. Moreover, each of them should only have two vertices
        while True:
            vertices = list(vertex_to_edges.keys())
            # Resolve numerical issues
            # Step 1: identify similar points:
            suspicous_vertices = self.get_suspicous_points(vertices)

            if len(suspicous_vertices) == 0:
                break

            updated = False
            edges = list(edge_to_vertices.keys())
            for e in edges:
                e_vs = edge_to_vertices.get(e)
                if len(e_vs) > 2:
                    continue
                all_suspicious = True
                for v in e_vs:
                    if v not in suspicous_vertices:
                        all_suspicious = False
                        break
                if not all_suspicious:
                    continue
                # remove e:
                updated = True
                removed_vs = []
                for v in e_vs:
                    if len(vertex_to_edges[v]) <= 2:
                        removed_vs.append(v)
                        vertex_to_edges.pop(v, None)

                edge_to_vertices.pop(e)

                for (e1, vs) in edge_to_vertices.items():
                    for v in removed_vs:
                        if v in vs:
                            vs.remove(v)

                break

            if not updated:
                break

        ret = list(vertex_to_edges)

        if len(self.get_suspicous_points(ret)) != 0:
            print("Warning")

        assert(len(self.get_suspicous_points(ret)) == 0)

        ret1 = [ret[0]]
        cur = ret[0]
        while True:
            edges = vertex_to_edges[cur]
            cur = None
            for e in edges:
                if cur is None:
                    for v in edge_to_vertices[e]:
                        if v not in ret1:
                            ret1.append(v)
                            cur = v
                            break
            if cur is None:
                break
        if (len(ret) != len(ret1)):
            U.verbose("Warning: ther emight be numerical issues.", 0)

        if len(ret) < 2:
            U.verbose("Warning: a region with just one vertex!",0)
            return ret

        # Final processing:
        # 1, make points clockwise
        ret = ret1
        sum_points = [0,0]
        for p in ret:
            sum_points[0] += p[0]
            sum_points[1] += p[1]
        sum_points[0] /= len(ret)
        sum_points[1] /= len(ret)

        mid_point = tuple(sum_points)

        point0 = ret[0]
        point1 = ret[1]

        vec0 = (point0[0] - mid_point[0], point0[1] - mid_point[1])
        vec1 = (point1[0] - mid_point[0], point1[1] - mid_point[1])

        # calculate cross_product
        cross = vec0[0] * vec1[1] - vec0[1]*vec1[0]

        # change it to clock wise
        if cross > 0:
            ret = list(reversed(ret))

        # 2. truncate floats

        for i in range(len(ret)):
            c_p = ret[i]
            n_p = (int(c_p[0] * U.LINEAR_TOLERANCE_SCALE) / float(U.LINEAR_TOLERANCE_SCALE),
                   int(c_p[1] * U.LINEAR_TOLERANCE_SCALE) / float(U.LINEAR_TOLERANCE_SCALE))
            ret[i] = n_p

        return ret


    '''
    Check whether a axis-aligned box intersects with any of the surfcace of current linear region.
    We use ubs and lbs to represent the bound on each dimension of the box.
    '''
    def checkBoxSurfaceIntersection(self, ubs, lbs):
        num_surfaces = len(self.ops)
        if self.sur_chk_model is None:
            self.sur_chk_model, self.sur_chk_vars, self.sur_chk_cons = self.constructGurobiModel()
            self.sur_chk_model.update()

        m, vs, cs = self.sur_chk_model, self.sur_chk_vars, self.sur_chk_cons
        # add constraints to encode the box
        assert(len(cs) == num_surfaces)

        # update the cache and do the real check
        for i in range(self.A.shape[0]):
            vs[i].ub = ubs[i]
            vs[i].lb = lbs[i]

        for i in range(num_surfaces):
            if self.internal_edges[i]:
                continue
            ci = cs[i]
            old_sense = ci.Sense
            ci.Sense = gurobipy.GRB.EQUAL
            m.update()
            m.optimize()
            s = m.status
            ci.Sense = old_sense
            m.update() # recover the sense
            if s != gurobipy.GRB.INFEASIBLE:
                return True

        return False


    '''
    Ideally, checking the intersection of two polytopes needs checking the intersections of all surfaces.
    For efficiency in high dimension, we use bounding boxes of surfaces to do the check.
    '''
    def getSurfaceBoxes(self):
        ret = []
        working_copy = self.copy()
        for i in range(len(self.ops)):
            a = self.A.T[i]
            b = self.B[i]
            old_op = working_copy.ops[i]
            if self.internal_edges[i]:
                continue
            working_copy.ops[i] = ExprOp.eq
            bounds = working_copy.get_bounds()
            if bounds is not None:
                (ubs, lbs) = bounds
                ret.append((ubs, lbs))
            working_copy.ops[i] = old_op
        return ret

    def markInternalSurface(self, i):
        self.internal_edges[i] = True

    '''
    Efficient implementation of copy
    '''
    def copy(self):
        ret_A = np.copy(self.A)
        ret_B = copy.copy(self.B)
        ret_C = copy.copy(self.ops)
        ret = LinearConstraintSet(ret_A, ret_B, ret_C)
        ret.internal_edges = copy.copy(self.internal_edges)
        return ret

class LinearRegion:
    def __init__(self):
        self.reluRecords = []
        self.linearCons = []
        self.additionalCons = None
        self.fixed_vars = None
        self.fixed_var_vals = None
        self.reset_bounds()
        self.pre_marked_redun = None

    def reset_bounds(self):
        self.UBs = None
        self.LBs = None
        self.redundant_cons = None

    def premark_redundant(self, idx, n_relu):
        if self.pre_marked_redun is None:
            self.pre_marked_redun = set()
        self.pre_marked_redun.add((idx, n_relu))

    def mark_redundant(self):
        self.redundant_cons = set()
        if self.LBs is None or self.UBs is None:
            bounds = self.getLinearConstrPre().get_bounds()
            if bounds is None:
                return None
            else:
                self.UBs, self.LBs = self.getLinearConstrPre().get_bounds()
        if self.fixed_vars is not None:
            num_vars = self.get_num_vars()
            nUBs = np.zeros(num_vars)
            nLBs = np.zeros(num_vars)
            nUBs[self.fixed_vars] = self.fixed_var_vals
            nLBs[self.fixed_vars] = self.fixed_var_vals
            nULBCounts = 0
            for i in range(num_vars):
                if i not in self.fixed_vars:
                    nLBs[i] = self.LBs[nULBCounts]
                    nUBs[i] = self.UBs[nULBCounts]
                    nULBCounts+=1
            self.UBs = nUBs
            self.LBs = nLBs

        # Note here the idx is different from the one in reluRecord.
        # Here the idx represents the nth relu layer, whereas there idx there represents nth layer.
        for (idx,c) in enumerate(self.linearCons):
            for n_relu in range(len(c.A.T)):
                a = c.A.T[n_relu]
                b = c.B[n_relu]
                op = c.ops[n_relu]
                # check if all 0s, trival case
                if all(v == 0 for v in a):
                    self.redundant_cons.add((idx, n_relu))
                    continue
                # Fast check using bounds:
                if op != ExprOp.eq:
                    sense = (op == ExprOp.ge)
                    lhs = b
                    for i,v in enumerate(a):
                        if (v >= 0 and sense) or (v < 0 and not sense):
                            lhs += self.LBs[i] * v
                        else:
                            lhs += self.UBs[i] * v
                    if (sense and lhs > 2 * U.LINEAR_TOLERANCE) or (not sense and lhs < -2 * U.LINEAR_TOLERANCE):
                        self.redundant_cons.add((idx, n_relu))

        return self.redundant_cons

    def addReluLayerConstraint(self, record, cons):
        self.reluRecords.append(record)
        self.linearCons.append(cons)
        self.reset_bounds()
    
    def addAddtionalConstraint(self, cons):
        if self.additionalCons is None:
            self.additionalCons = cons
        else:
            self.additionalCons = self.additionalCons.extend(cons)
        self.reset_bounds()

    '''
    Fix the values of certain variables.
    This can be done by introducing new constraints. But Gurobi is slow in adding constraints. So removing is better.
    '''
    def fix_vars(self, var_list, val_list):
        self.fixed_vars = var_list
        self.fixed_var_vals = val_list
        self.reset_bounds()

    def getLinearConstr(self):
        if self.fixed_vars is not None:
            return self.getLinearConstrSimplified()
        return self.getLinearConstrPre()

    def getLinearConstrRaw(self):
        cons = None
        for c in self.linearCons:
            if cons is None:
                cons = c
            else:
                cons = cons.extend(c)
        if self.additionalCons is not None:
            cons = cons.extend(self.additionalCons)

        if self.fixed_vars is not None:
            cons = cons.fix_vars(self.fixed_vars, self.fixed_var_vals)

        return cons

    def getLinearConstrPre(self):
        if self.pre_marked_redun is None:
            return self.getLinearConstrRaw()
        retAT = []
        retB = []
        retOps = []
        for (idx, c) in enumerate(self.linearCons):
            AT = c.A.T
            for ridx in range(len(c.B)):
                if (idx, ridx) not in self.pre_marked_redun:
                    retAT.append(AT[ridx])
                    retB.append(c.B[ridx])
                    retOps.append(c.ops[ridx])

        ret = LinearConstraintSet(np.transpose(retAT), np.array(retB), np.array(retOps))
        if self.additionalCons is not None:
            ret = ret.extend(self.additionalCons)

        if self.fixed_vars is not None:
            ret = ret.fix_vars(self.fixed_vars, self.fixed_var_vals)

        return ret

    def getLinearConstrSimplified(self):
        if self.redundant_cons is None:
            if self.mark_redundant() is None:
                return None
        retAT = []
        retB = []
        retOps = []
        for (idx, c) in enumerate(self.linearCons):
            AT = c.A.T
            for ridx in range(len(c.B)):
                if ((idx, ridx) not in self.redundant_cons)\
                        and (self.pre_marked_redun is None or (idx, ridx) not in self.pre_marked_redun):
                    retAT.append(AT[ridx])
                    retB.append(c.B[ridx])
                    retOps.append(c.ops[ridx])

        ret = LinearConstraintSet(np.transpose(retAT), np.array(retB), np.array(retOps))
        if self.additionalCons is not None:
            ret = ret.extend(self.additionalCons)

        if self.fixed_vars is not None:
            ret = ret.fix_vars(self.fixed_vars, self.fixed_var_vals)

        return ret


    def check(self):
        cons = self.getLinearConstr()
        if cons is None:
            return False
        return cons.solve() is not None

    def get_num_vars(self):
        return len(self.linearCons[0].A)

    """
    check if the additional constraint holds on a given boundary of the linear region.
    We use n_layer and n_relu to locate the boundary.
    n_layer: the index of the relu layer among all relu layers
    n_relu: the index of the relu in the layer
    """
    def checkBoundary(self, n_layer, n_relu):
        if self.additionalCons is None:
            raise ValueError('No additional constraints on this linear region.')


        if self.redundant_cons is None:
            self.mark_redundant()

        if (n_layer, n_relu) in self.redundant_cons \
                or (self.pre_marked_redun is not None and (n_layer, n_relu) in self.pre_marked_redun):
            return False

        cons = None
        for (idx,c) in enumerate(self.linearCons):
            if n_layer == idx:
                a = c.A.T[n_relu]
                b = c.B[n_relu]
                op = c.ops[n_relu]

                c = c.copy()
                c.ops[n_relu] = ExprOp.eq
            if cons is None:
                cons = c.copy()
            else:
                cons = cons.extend(c)
        cons = cons.extend(self.additionalCons)
        if self.fixed_vars is not None:
            cons = cons.fix_vars(self.fixed_vars, self.fixed_var_vals)
        return cons.solve() is not None

    '''
    Mark a certain linear constraint as internal to the polytope
    n_layer: the position among all ReLU layers
    '''
    def markInternalSurface(self, n_layer, n_relu):
        self.linearCons[n_layer].markInternalSurface(n_relu)

    def copy(self):
        ret = LinearRegion()
        ret.reluRecords = self.reluRecords[:]
        ret.linearCons = self.linearCons[:]
        ret.additionalCons = self.additionalCons[:]
        ret.fixed_vars = self.fixed_vars
        ret.fixed_var_vals = self.fixed_var_vals
        ret.pre_marked_redun = self.pre_marked_redun
        return ret

    def __eq__(self, o: object) -> bool:
        return self.__dict__ == o.__dict__


'''
The main class that captures the piece-wise linear function represented by the neural network.
Note that the support for maxpool is not complete yet.
Conceptually, it is the same as handling relu.
But things just get tricky engineering-wise given problems like padding, strides, and channels.
'''
class Model:
    def __init__(self, layerList, input_length, mean = None, std = None):
        self.layerList = layerList
        self.mean = mean
        self.std = std
        self.input_length = input_length
        self.redundant_relus = None
    
    def getNumLayers(self):
        return len(self.layerList)
    
    def getInputVariable(self):
        inputLayer = self.layerList[0]
        assert(inputLayer.kind == LayerKind.input)
        return inputLayer.var
    
    def getOutputVariable(self):
        depth = self.getNumLayers() - 1
        outputLayer = self.layerList[depth - 1]
        return outputLayer.var
    
    def getProbVar(self):
        return tf.nn.softmax(self.getOutputVariable)

    def _get_conv_output_length(self, in_seq_length, fw, stride, padding):
        if not padding:
            out_seq_length = in_seq_length - fw + 1
            out_seq_length = out_seq_length / stride
        else:
            out_seq_length = in_seq_length / stride
        return int(out_seq_length)

    def _get_conv_input_length(self, last_layer_num, num_in_channels):
        in_seq_length = last_layer_num / num_in_channels

        return int(in_seq_length)


    def _encode_conv1d_bias_matrix(self, last_layer_num, filter_weights, filter_biases, stride=1, padding=True):
        filter_shape = np.shape(filter_weights)
        num_in_channels = filter_shape[1]
        num_out_channels = len(filter_biases)

        in_seq_length = self._get_conv_input_length(last_layer_num, num_in_channels)

        fw = filter_shape[0]

        out_seq_length = self._get_conv_output_length(in_seq_length, fw, stride, padding)

        fb_matrix = np.zeros([out_seq_length * num_out_channels])

        for i in range(out_seq_length):
            for j in range(num_out_channels):
                fb_matrix[i*num_out_channels + j] = filter_biases[j]

        return fb_matrix

    def _encode_conv1d_weight_matrix(self, last_layer_num, filter_weights, stride=1, padding=True):
        filter_shape = np.shape(filter_weights)
        num_in_channels = filter_shape[1]
        num_out_channels = filter_shape[2]

        in_seq_length = self._get_conv_input_length(last_layer_num, num_in_channels)

        fw = filter_shape[0]
        half_fw = int(filter_shape[0] /2)

        out_seq_length = self._get_conv_output_length(in_seq_length, fw, stride, padding)

        fw_matrix = np.zeros([int(in_seq_length * num_in_channels), int(out_seq_length * num_out_channels)])

        # Let's fill in the matrix
        # Scan the number of output pixels
        for i in range(out_seq_length):
            # Scan the number of channels
            for j in range(num_out_channels):
                # get the weights
                fw_slice = filter_weights[:,:,j]
                fw_slice = np.ndarray.flatten(fw_slice)
                # fill in the weights

                for k in range(filter_shape[0]):
                    if padding:
                        idx = i*stride - half_fw + k
                    else:
                        idx = i*stride + k
                    if idx >= 0 and idx < in_seq_length:
                        for l in range(num_in_channels):
                            fw_matrix[idx*num_in_channels+l,i*num_out_channels+j] = fw_slice[k*num_in_channels+l]

        return fw_matrix


    def _encode_maxpool_matrix(self, X, A, b, num_channels = 1, window_size = 1, stride = 1, padding = True):
        last_lay_num = self.input_length
        if A is not None:
            last_lay_num = np.shape(A)[1]
        in_seq_length = self._get_conv_input_length(last_lay_num, num_channels)
        out_seq_length = self._get_conv_output_length(in_seq_length, window_size, stride, padding)

        cur_val = np.dot(X, A)
        cur_val = np.add(cur_val, b)

        assert(len(np.shape(cur_val)) == 1)

        pool_matrix = np.zeros(int(in_seq_length*num_channels), int(out_seq_length*num_channels))

        hw = int(window_size/2)

        for i in range(out_seq_length):

            if padding:
                start_idx = i*stride - hw
            else:
                start_idx = i *stride

            for k in range(num_channels):
                # figure out the pool indices
                max_val = -np.infty
                m_idx = -1
                for j in range(window_size):
                    cur_idx = start_idx+j
                    if cur_idx >= 0 and cur_idx <  in_seq_length:
                        val = cur_val[cur_idx * num_channels + k]
                        if val > max_val:
                            max_val = val
                            m_id = cur_idx

                pool_matrix[m_idx * num_channels+k, i*num_channels+k] = 1

        return pool_matrix


    def _update_A_b(self, A, b, nA, nb):
        if A is None:
            A = copy.copy(nA)
            assert (b is None)
            b = copy.copy(nb)
        else:
            A = np.dot(A, nA)
            b = np.dot(b, nA)
            if nb is not None:
                b = np.add(b, nb)
        return (A,b)

    def _update_diff_A(self, A):
        assert(A is not None)
        assert(A.shape[0] == A.shape[1])
        A = copy.deepcopy(A)
        for i in range(A.shape[0]):
            A[i, i] = A[i, i]-1
        return A

    '''
    Return the symbolic expression of the output layer that X falls into
    '''
    def getOutLayerSymFromInput(self, X):
        num_channels = 1
        if len(np.shape(X)) > 1:
            num_channels = np.shape(X)[-1]
        #Let's flatten everything
        X = np.ndarray.flatten(X)

        A = None
        b = None
        for layer in self.layerList:
            if layer.kind == LayerKind.dense:
                A, b = self._update_A_b(A, b, layer.weights, layer.biases)
            elif layer.kind == LayerKind.relu:
                matrix = np.array([X])
                result = np.dot(matrix,A)
                result  = result + b
                assert(len(result.shape) == 2)
                for (idx, v) in enumerate(result[0]):
                    if v < 0:
                        A[:, idx] = 0
                        b[idx] = 0
            elif layer.kind == LayerKind.conv:
                # judge whether it is conv1d or conv2d
                filter_weights = layer.weights
                filter_biases = layer.biases
                padding = layer.padding_type
                stride = layer.strides
                num_channels = filter_weights.shape[-1]

                if len(filter_weights.shape) == 4: #2d conv
                    raise ValueError('2D convolution not supported yet!')
                if len(filter_weights.shape) == 3: #1d conv
                    last_layer_num = self.input_length
                    if A is not None:
                        last_layer_num = np.shape(A)[1]
                    fw_matrix = self._encode_conv1d_weight_matrix(last_layer_num, filter_weights, stride, padding == 'SAME')
                    fb_matrix = self._encode_conv1d_bias_matrix(last_layer_num, filter_weights, filter_biases, stride, padding == 'SAME')
                    A, b = self._update_A_b(A, b, fw_matrix, fb_matrix)
                else:
                    raise ValueError('Unsupported dimentions: '+str(len(filter_weights.shape) - 2))

            elif layer.kind == LayerKind.pool:
                pool_matrix = self._encode_maxpool_matrix(X, A, b, num_channels, window_size = layer.weights[0], stride = layer.strides[0], padding = (layer.padding_type == 'SAME'))
                A, b = self._update_A_b(A, b, pool_matrix, None)
                raise ValueError('The support for maxpool is not complete yet.')
            elif layer.kind == LayerKind.dropout:
                pass
            elif layer.kind == LayerKind.diff_input:
                A = self._update_diff_A(A)
            elif layer.kind == LayerKind.absolute:
                matrix = np.array([X])
                result = np.dot(matrix,A)
                result  = result + b
                assert(len(result.shape) == 2)
                for (idx, v) in enumerate(result[0]):
                    if v < 0:
                        A[:, idx] = -A[:, idx]
                        b[idx] = -b[idx]
            elif layer.kind == LayerKind.reshape:
                pass
            else:
                raise ValueError("Unsupported layer type: "+str(layer.kind))
        return LinearExpressions(A,b)


    '''
    Get the record of the activation functions.
    Note this alone is not enough to fix the linear piece if maxpool layer is present.
    But I am too lazy to handle maxpool for now.
    '''
    def getReluActivationRecord(self, X):
        num_channels = 1
        if len(np.shape(X)) > 1:
            num_channels = np.shape(X)[-1]
        #Let's flatten everything
        X = np.ndarray.flatten(X)

        A = None
        b = None
        ret = []
        for (idx, layer) in enumerate(self.layerList):
            if layer.kind == LayerKind.dense:
                if A is None:
                    A = copy.copy(layer.weights)
                    assert(b is None)
                    b = copy.copy(layer.biases)
                else:
                    A = np.dot(A,layer.weights)
                    b = np.dot(b,layer.weights)
                    b = np.add(b, layer.biases)
            elif layer.kind == LayerKind.relu:
                record = []
                # rec = ReluRecord(idx, [])
                matrix = np.array([X])
                result = np.dot(matrix,A)
                result = result + b
                assert(len(result.shape) == 2)
                for (idx1, v) in enumerate(result[0]):
                    if v < 0:
                        A[:, idx1] = 0
                        b[idx1] = 0
                        record.append(0)
                    else:
                        record.append(1)
                rec = ReluRecord(idx, tuple(record))
                ret.append(rec)
            elif layer.kind == LayerKind.conv:
                # judge whether it is conv1d or conv2d
                filter_weights = layer.weights
                filter_biases = layer.biases
                padding = layer.padding_type
                stride = layer.strides
                num_channels = filter_weights.shape[-1]

                if len(filter_weights.shape) == 4: #2d conv
                    raise ValueError('2D convolution not supported yet!')
                if len(filter_weights.shape) == 3: #1d conv
                    last_layer_num = self.input_length
                    if A is not None:
                        last_layer_num = np.shape(A)[1]
                    fw_matrix = self._encode_conv1d_weight_matrix(last_layer_num, filter_weights, stride, padding == 'SAME')
                    fb_matrix = self._encode_conv1d_bias_matrix(last_layer_num, filter_weights, filter_biases, stride, padding == 'SAME')
                    A, b = self._update_A_b(A, b, fw_matrix, fb_matrix)
                else:
                    raise ValueError('Unsupported dimentions: '+str(len(filter_weights.shape) - 2))
            elif layer.kind == LayerKind.pool:
                pool_matrix = self._encode_maxpool_matrix(X, A, b, num_channels, window_size = layer.weights[0], stride = layer.strides[0], padding = (layer.padding_type == 'SAME'))
                A, b = self._update_A_b(A, b, pool_matrix, None)
                raise ValueError('The support for maxpool is not complete yet.')
            elif layer.kind == LayerKind.dropout:
                pass
            elif layer.kind == LayerKind.diff_input:
                A = self._update_diff_A(A)
            elif layer.kind == LayerKind.absolute:
                matrix = np.array([X])
                result = np.dot(matrix,A)
                result  = result + b
                assert(len(result.shape) == 2)
                # Here we treat the absolute layer as a special relu layer
                record = []
                for (idx1, v) in enumerate(result[0]):
                    if v < 0:
                        A[:, idx] = -A[:, idx]
                        b[idx] = -b[idx]
                        record.append(0)
                    else:
                        record.append(1)
                rec = ReluRecord(idx, tuple(record))
                ret.append(rec)
            elif layer.kind == LayerKind.reshape:
                pass
            else:
                raise ValueError("Unsupported layer kind: "+str(layer.kind))
        return ret
        
    '''
    Return the symbolic expression of the output layer that corresponds to a give relu record
    '''
    def getOutLayerSymFromRecords(self, reluRecords):
        A = None
        b = None
        for (idx, layer) in enumerate(self.layerList):
            if layer.kind == LayerKind.dense:
                if A is None:
                    A = copy.copy(layer.weights)
                    assert(b is None)
                    b = copy.copy(layer.biases)
                else:
                    A = np.dot(A,layer.weights)
                    b = np.dot(b,layer.weights)
                    b = np.add(b, layer.biases)
            elif layer.kind == LayerKind.relu:
                for r in reluRecords:
                    if r.idx == idx:
                        rec = r
                        break
                    
                for (idx, v) in enumerate(rec.record):
                    if v == 0:
                        A[:, idx] = 0
                        b[idx] = 0
            elif layer.kind == LayerKind.conv:
                # judge whether it is conv1d or conv2d
                filter_weights = layer.weights
                filter_biases = layer.biases
                padding = layer.padding_type
                stride = layer.strides
                num_channels = filter_weights.shape[-1]

                if len(filter_weights.shape) == 4: #2d conv
                    raise ValueError('2D convolution not supported yet!')
                if len(filter_weights.shape) == 3: #1d conv
                    last_layer_num = self.input_length
                    if A is not None:
                        last_layer_num = np.shape(A)[1]
                    fw_matrix = self._encode_conv1d_weight_matrix(last_layer_num, filter_weights, stride, padding == 'SAME')
                    fb_matrix = self._encode_conv1d_bias_matrix(last_layer_num, filter_weights, filter_biases, stride, padding == 'SAME')
                    A, b = self._update_A_b(A, b, fw_matrix, fb_matrix)
                else:
                    raise ValueError('Unsupported dimentions: '+str(len(filter_weights.shape) - 2))
            elif layer.kind == LayerKind.pool:
                # pool_matrix = self.encode_maxpool_matrix(X, A, b, num_channels, window_size = layer.weights[0], stride = layer.strides[0], padding = (layer.padding_type == 'SAME'))
                # A, b = self.update_A_b(A, b, pool_matrix, None)
                raise ValueError('The support for maxpool is not complete yet.')
            elif layer.kind == LayerKind.dropout:
                pass
            elif layer.kind == LayerKind.diff_input:
                A = self._update_diff_A(A)
            elif layer.kind == LayerKind.absolute:
                for r in reluRecords:
                    if r.idx == idx:
                        rec = r
                        break
                    
                for (idx, v) in enumerate(rec.record):
                    if v == 0:
                        A[:, idx] = -A[:, idx]
                        b[idx] = -b[idx]        
            elif layer.kind == LayerKind.reshape:
                pass
            else:
                raise ValueError('Unsupported layer kind: '+str(layer.kind))
        return LinearExpressions(A,b)

    def getLinearRegionFromInput(self, X):
        ret = LinearRegion()
        A = None
        b = None
        for (idx, layer) in enumerate(self.layerList):
            if layer.kind == LayerKind.dense:
                if A is None:
                    A = copy.copy(layer.weights)
                    assert(b is None)
                    b = copy.copy(layer.biases)
                else:
                    A = np.dot(A,layer.weights)
                    b = np.dot(b,layer.weights)
                    b = np.add(b, layer.biases)
            elif layer.kind == LayerKind.relu:
                ops = []
                record = []
                # rec = ReluRecord(idx, [])
                matrix = np.array([X])
                result = np.dot(matrix,A)
                result = result + b
                assert(len(result.shape) == 2)
            
                A1 = np.copy(A)
                b1 = np.copy(b)
                
                for (idx1, v) in enumerate(result[0]):
                    if v < 0:
                        A[:, idx1] = 0
                        b[idx1] = 0
                        record.append(0)
                        ops.append(ExprOp.le)
                    else:
                        record.append(1)
                        ops.append(ExprOp.ge)
                rec = ReluRecord(idx, tuple(record))
                ret.addReluLayerConstraint(rec, LinearConstraintSet(A1, b1, ops))
            elif layer.kind == LayerKind.conv:
                # judge whether it is conv1d or conv2d
                filter_weights = layer.weights
                filter_biases = layer.biases
                padding = layer.padding_type
                stride = layer.strides
                num_channels = filter_weights.shape[-1]

                if len(filter_weights.shape) == 4: #2d conv
                    raise ValueError('2D convolution not supported yet!')
                if len(filter_weights.shape) == 3: #1d conv
                    last_layer_num = self.input_length
                    if A is not None:
                        last_layer_num = np.shape(A)[1]
                    fw_matrix = self._encode_conv1d_weight_matrix(last_layer_num, filter_weights, stride, padding == 'SAME')
                    fb_matrix = self._encode_conv1d_bias_matrix(last_layer_num, filter_weights, filter_biases, stride, padding == 'SAME')
                    A, b = self._update_A_b(A, b, fw_matrix, fb_matrix)
                else:
                    raise ValueError('Unsupported dimentions: '+str(len(filter_weights.shape) - 2))
            elif layer.kind == LayerKind.pool:
                pool_matrix = self._encode_maxpool_matrix(X, A, b, num_channels, window_size = layer.weights[0], stride = layer.strides[0], padding = (layer.padding_type == 'SAME'))
                A, b = self._update_A_b(A, b, pool_matrix, None)
                raise ValueError('The support for maxpool is not complete yet.')
            elif layer.kind == LayerKind.dropout:
                pass
            elif layer.kind == LayerKind.diff_input:
                A = self._update_diff_A(A)
            elif layer.kind == LayerKind.absolute:
                ops = []
                record = []
                # rec = ReluRecord(idx, [])
                matrix = np.array([X])
                result = np.dot(matrix,A)
                result = result + b
                assert(len(result.shape) == 2)
            
                A1 = np.copy(A)
                b1 = np.copy(b)
                
                for (idx1, v) in enumerate(result[0]):
                    if v < 0:
                        A[:, idx] = -A[:, idx]
                        b[idx] = -b[idx]
                        record.append(0)
                        ops.append(ExprOp.le)
                    else:
                        record.append(1)
                        ops.append(ExprOp.ge)
                rec = ReluRecord(idx, tuple(record))
                ret.addReluLayerConstraint(rec, LinearConstraintSet(A1, b1, ops))
            elif layer.kind == LayerKind.reshape:
                pass
            else:
                raise ValueError('Unsupported layer kind: '+str(layer.kind))
        if self.redundant_relus is not None:
            for (idx, relu_n) in self.redundant_relus:
                ret.premark_redundant(idx, relu_n)
        return ret

    def getLinearRegionFromRecord(self, records: List[ReluRecord]):
        ret = LinearRegion()
        A = None
        b = None
        for (idx, layer) in enumerate(self.layerList):
            if layer.kind == LayerKind.dense:
                if A is None:
                    A = copy.copy(layer.weights)
                    assert (b is None)
                    b = copy.copy(layer.biases)
                else:
                    A = np.dot(A, layer.weights)
                    b = np.dot(b, layer.weights)
                    b = np.add(b, layer.biases)
            elif layer.kind == LayerKind.relu:
                ops = []
                rec : ReluRecord = None
                for r in records:
                    if r.idx == idx:
                        rec = r
                        break

                A1 = np.copy(A)
                b1 = np.copy(b)

                for (idx1, v) in enumerate(rec.record):
                    if v == 0:
                        A[:, idx1] = 0
                        b[idx1] = 0
                        ops.append(ExprOp.le)
                    else:
                        ops.append(ExprOp.ge)
                ret.addReluLayerConstraint(rec, LinearConstraintSet(A1, b1, ops))
            elif layer.kind == LayerKind.conv:
                # judge whether it is conv1d or conv2d
                filter_weights = layer.weights
                filter_biases = layer.biases
                padding = layer.padding_type
                stride = layer.strides
                num_channels = filter_weights.shape[-1]

                if len(filter_weights.shape) == 4: #2d conv
                    raise ValueError('2D convolution not supported yet!')
                if len(filter_weights.shape) == 3: #1d conv
                    last_layer_num = self.input_length
                    if A is not None:
                        last_layer_num = np.shape(A)[1]
                    fw_matrix = self._encode_conv1d_weight_matrix(last_layer_num, filter_weights, stride, padding == 'SAME')
                    fb_matrix = self._encode_conv1d_bias_matrix(last_layer_num, filter_weights, filter_biases, stride, padding == 'SAME')
                    A, b = self._update_A_b(A, b, fw_matrix, fb_matrix)
                else:
                    raise ValueError('Unsupported dimentions: '+str(len(filter_weights.shape) - 2))
            elif layer.kind == LayerKind.pool:
                # pool_matrix = self.encode_maxpool_matrix(X, A, b, num_channels, window_size = layer.weights[0], stride = layer.strides[0], padding = (layer.padding_type == 'SAME'))
                # A, b = self.update_A_b(A, b, pool_matrix, None)
                raise ValueError('The support for maxpool is not complete yet.')
            elif layer.kind == LayerKind.dropout:
                pass
            elif layer.kind == LayerKind.diff_input:
                A = self._update_diff_A(A)
            elif layer.kind == LayerKind.absolute:
                ops = []
                rec : ReluRecord = None
                for r in records:
                    if r.idx == idx:
                        rec = r
                        break

                A1 = np.copy(A)
                b1 = np.copy(b)

                for (idx1, v) in enumerate(rec.record):
                    if v == 0:
                        A[:, idx] = -A[:, idx]
                        b[idx] = -b[idx]
                        ops.append(ExprOp.le)
                    else:
                        ops.append(ExprOp.ge)
                ret.addReluLayerConstraint(rec, LinearConstraintSet(A1, b1, ops))                
            elif layer.kind == LayerKind.reshape:
                pass
            else:
                raise ValueError('Unsupported layer kind '+str(layer.kind))
        if self.redundant_relus is not None:
            for (idx, relu_n) in self.redundant_relus:
                ret.premark_redundant(idx, relu_n)
        return ret

    def getNumReLUs(self):
        ret = 0
        lastNum = self.input_length
        for layer in self.layerList:
            if layer.kind == LayerKind.relu:
                ret += lastNum
            if layer.kind == LayerKind.dense:
                lastNum = len(layer.biases)
            if layer.kind == LayerKind.conv:
                filter_shape = np.shape(layer.weights)
                num_in_channels = filter_shape[1]
                num_out_channels = filter_shape[2]
                in_seq_length = lastNum / num_in_channels

                fw = filter_shape[0]

                out_seq_length = self._get_conv_output_length(in_seq_length, fw, layer.strides, layer.padding_type == 'SAME')
                lastNum = out_seq_length * num_out_channels
            if layer.kind == LayerKind.pool:
                raise ValueError('Maxpool not supported yet!')
        return ret

    # Simple 
    def analyzeRedundantDead(self):
        ret = 0
        lastNum = self.input_length
        lastZeros = []
        n_relu_layer = 0
        for layer in self.layerList:
            # We assume a ReLu layer always follows a dense layer
            if layer.kind == LayerKind.relu:
                ret += len(lastZeros)
                for idx in lastZeros:
                    self.redundant_relus.add((n_relu_layer, idx))
                n_relu_layer += 1

            if layer.kind == LayerKind.dense:
                fw_matrix = layer.weights
                fb_matrix = layer.biases

            if layer.kind == LayerKind.conv:
                filter_weights = layer.weights
                filter_biases = layer.biases
                stride = layer.strides
                padding = layer.padding_type
                fw_matrix = self._encode_conv1d_weight_matrix(lastNum, filter_weights, stride, padding == 'SAME')
                fb_matrix = self._encode_conv1d_bias_matrix(lastNum, filter_weights, filter_biases, stride,
                                                           padding == 'SAME')
            if layer.kind == LayerKind.dense or layer.kind == LayerKind.conv:
                lastZeros = []
                for idx, (a, b) in enumerate(zip(fw_matrix.T, fb_matrix)):
                    if (not np.any(a)):
                        lastZeros.append(idx)
                lastNum = len(fb_matrix)

            if layer.kind == LayerKind.pool:
                raise ValueError('Maxpool not supported yet!')
            
            if layer.kind == LayerKind.diff_input:
                pass
            if layer.kind == LayerKind.absolute:
                pass
                
        return ret

    def analyzeRedundantInterval(self, x_val, change_indices, bound_map):
        intervalMap = {} # dictionary that maps the output of last layer to its lb and ub
        ret = 0
        lastNum = self.input_length
        n_relu_layers = 0
        for i in range(len(x_val)):
            if i not in change_indices:
                intervalMap[i] = [x_val[i], x_val[i]]
            else:
                intervalMap[i] = bound_map[i]

        for layer in self.layerList:
            if layer.kind == LayerKind.relu:
                lastN = 0
                change_list = []
                for k,v in intervalMap.items():
                    if v[0] >= 0:
                        lastN += 1
                        self.redundant_relus.add((n_relu_layers, k))
                    if v[1] < 0:
                        lastN += 1
                        self.redundant_relus.add((n_relu_layers, k))
                        change_list.append([k, [0, 0]])
                for k,v in change_list:
                    intervalMap[k] = v
                ret += lastN
                n_relu_layers += 1

            if layer.kind == LayerKind.dense:
                fw_matrix = layer.weights
                fb_matrix = layer.biases

            if layer.kind == LayerKind.conv:
                filter_weights = layer.weights
                filter_biases = layer.biases
                stride = layer.strides
                padding = layer.padding_type
                fw_matrix = self._encode_conv1d_weight_matrix(lastNum, filter_weights, stride, padding == 'SAME')
                fb_matrix = self._encode_conv1d_bias_matrix(lastNum, filter_weights, filter_biases, stride,
                                                           padding == 'SAME')
            if layer.kind == LayerKind.dense or layer.kind == LayerKind.conv:
                n_intervalMap = {}
                for idx, (a, b) in enumerate(zip(fw_matrix.T, fb_matrix)):
                    ub = b
                    lb = b
                    for aidx, acoe in enumerate(a):
                        if acoe > 0:
                            ub += acoe * intervalMap[aidx][1]
                            lb += acoe * intervalMap[aidx][0]
                        else:
                            ub += acoe * intervalMap[aidx][0]
                            lb += acoe * intervalMap[aidx][1]
                        n_intervalMap[idx] = [lb, ub]
                lastNum = len(fb_matrix)
                intervalMap = n_intervalMap

            if layer.kind == LayerKind.pool:
                print('Maxpool not supported yet!')
                break
            
            if layer.kind == LayerKind.absolute:
                print('Absolute not supported yet!')
                break
            
            if layer.kind == LayerKind.diff_input:
                print('input_diff not supported yet!')
                break
        return ret

    def analyzeRedundant(self, x_val, change_indices, bound_map):
        self.redundant_relus = set()
        U.verbose('Number of reLUs: '+str(self.getNumReLUs()), 0)
        rm1 = self.analyzeRedundantDead()
        U.verbose('Number of dead reLUs: '+str(rm1), 0)

        rm2 = self.analyzeRedundantInterval(x_val, change_indices, bound_map)
        U.verbose('Number of reLUs removed by the analysis: '+str(rm2), 0)

        # bound_map = {}
        # change_indices = []
        # for i in range(len(x_val)):
        #     bound_map[i] = [0, 1]
        #     change_indices.append(i)
        # rm3 = self.analyzeRedundantInterval(x_val, change_indices, bound_map)
        # verbose('Number of reLUs removed by the analysis: '+str(rm3), 0)
