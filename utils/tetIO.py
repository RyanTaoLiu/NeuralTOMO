import numpy as np
import pyvista as pv
import tetgen


def getVolumeMeshBoundary(elem):
    FacesDict = dict()
    nodeSet = set()
    FacesIndex = list()
    Face2Elem = dict()
    FacePointsIdx = dict()
    Face2ElemIndex = list()

    for idx, it in enumerate(elem):
        '''
        F_{1,2,3,4}:[0,1,2],[0,2,3],[0,1,3],[1,3,2]
        '''
        F_0 = np.array([it[0], it[2], it[1]], dtype=int)
        F_1 = np.array([it[0], it[3], it[2]], dtype=int)
        F_2 = np.array([it[0], it[1], it[3]], dtype=int)
        F_3 = np.array([it[1], it[2], it[3]], dtype=int)

        F_0_sorted = np.sort(F_0)
        F_1_sorted = np.sort(F_1)
        F_2_sorted = np.sort(F_2)
        F_3_sorted = np.sort(F_3)

        s0 = '{}_{}_{}'.format(F_0_sorted[0], F_0_sorted[1], F_0_sorted[2])
        s1 = '{}_{}_{}'.format(F_1_sorted[0], F_1_sorted[1], F_1_sorted[2])
        s2 = '{}_{}_{}'.format(F_2_sorted[0], F_2_sorted[1], F_2_sorted[2])
        s3 = '{}_{}_{}'.format(F_3_sorted[0], F_3_sorted[1], F_3_sorted[2])

        FacePointsIdx[s0] = F_0
        FacePointsIdx[s1] = F_1
        FacePointsIdx[s2] = F_2
        FacePointsIdx[s3] = F_3

        L = [s0, s1, s2, s3]
        for s in L:
            Face2Elem[s] = idx
            if s in FacesDict:
                FacesDict[s] += 1
            else:
                FacesDict[s] = 0

    for vit, fit in FacesDict.items():
        if fit == 0:
            Face2ElemIndex.append(Face2Elem[vit])
            lvit = FacePointsIdx[vit]
            # lvit = vit.split('_')

            for lvit_it in lvit:
                nodeSet.add(int(lvit_it))
                # FacesIndex.append(int(lvit_it))
                FacesIndex.append(int(lvit_it))

    nodesIndex = np.array(list(nodeSet))
    nodesIndex.sort()
    FacesIndex = np.array(FacesIndex).reshape(-1, 3)

    nodesIndexInverse = np.zeros(nodesIndex.flatten().max() + 1, dtype=int) - 1
    for i in range(nodesIndex.shape[0]):
        nodesIndexInverse[nodesIndex[i]] = i

    FacesIndex = nodesIndexInverse[FacesIndex]
    assert (FacesIndex.flatten().min() > -1)

    return nodesIndex, FacesIndex, Face2ElemIndex


def toTetgenCell(node, elem):
    cells = np.hstack((np.zeros((elem.shape[0], 1), dtype=int) + 4, elem))
    cell_type = np.zeros(cells.shape[0], dtype="uint8") + pv.CellType.TETRA
    grid = pv.UnstructuredGrid(cells, cell_type, node)
    return grid


def loadTet(filePath):
    with open(filePath) as f:
        numofVertices = int(f.readline().split(' ')[0])
        numofEles = int(f.readline().split(' ')[0])
        v, e = list(), list()
        for i in range(numofVertices):
            slist = f.readline().split(' ')
            v.append([float(slist[0]), float(slist[1]), float(slist[2])])
        for j in range(numofEles):
            elist = f.readline().split(' ')
            e.append([int(elist[1]), int(elist[2]), int(elist[3]), int(elist[4])])
        mesh = toTetgenCell(np.asarray(v, dtype=float), np.asarray(e, dtype=int))
        return mesh


def saveTet(fliePath, node, elem):
    with open(fliePath, "w") as f:
        f.write('{} vertices\n{} tets\n'.format(node.shape[0], elem.shape[0]))
        for i in range(node.shape[0]):
            f.write('{} {} {}\n'.format(node[i][0], node[i][1], node[i][2]))
        for i in range(elem.shape[0]):
            f.write('{} {} {} {} {}\n'.format(4, elem[i][0], elem[i][1], elem[i][2], elem[i][3]))


def getNormal(v, f):
    VF = v[f]  # Nx3x3
    n = np.cross(VF[:, 1, :] - VF[:, 0, :], VF[:, 2, :] - VF[:, 0, :])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def tetrahedron_generate_from_mesh(mesh, make_manifold=True, verbose=False):
    sourceOrgMeshTet = tetgen.TetGen(mesh)
    if make_manifold:
        sourceOrgMeshTet.make_manifold(verbose=verbose)
    boundaryV, boundaryF = sourceOrgMeshTet.v.copy(), sourceOrgMeshTet.f.copy()
    # node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.5/60a{}Y".format(100), verbose=verbose)
    node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.2/10a{}Y".format(10000), verbose=verbose)
    # node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.2/10a{}Y".format(0.1), verbose=verbose)
    print("[Tetrahedral]: Number of Nodes {}, Number of Elements {}".format(node.shape[0], elem.shape[0]))
    return node, elem, sourceOrgMeshTet.grid, boundaryV, boundaryF


def initScalar2GradientMesh(node, elem, weight):
    mesh_elem, mesh_node = elem, node
    elem_nodes = mesh_node[mesh_elem]  # nx4x3

    # to calculate the gradient of every element
    # knows the vertices and weights of every elem
    # get the gradient nx3
    faces_idx = np.asarray([[0, 1, 2],
                            [0, 3, 1],
                            [0, 2, 3],
                            [1, 3, 2]],
                           dtype=int)
    # first h = (v1-v0) \cross (v2-v0)
    e1 = elem_nodes[:, 1, :] - elem_nodes[:, 0, :]
    e2 = elem_nodes[:, 2, :] - elem_nodes[:, 0, :]
    e3 = elem_nodes[:, 3, :] - elem_nodes[:, 0, :]
    vol = (e3 * np.cross(e1, e2)).sum(1) / 6

    grad = np.zeros((mesh_elem.shape[0], 4, 3))

    for i in range(4):
        face = faces_idx[i]
        e1 = elem_nodes[:, face[1], :] - elem_nodes[:, face[0], :]
        e2 = elem_nodes[:, face[2], :] - elem_nodes[:, face[0], :]
        h = np.cross(e1, e2)
        # height[:, i, :] = torch.nn.functional.normalize(h) * (vol/h.norm(dim=1)).unsqueeze(-1).expand(-1,3)
        grad[:, i, :] = h / np.expand_dims((vol * 3), axis=-1).repeat(3, -1) / 2

    elem_weight = weight[mesh_elem]  # nx4
    faces_opposite_node_idx = np.array([3, 2, 1, 0], dtype=int)

    # first h = (v1-v0) \cross (v2-v0)
    elem_gradient = np.zeros((mesh_elem.shape[0], 3))
    for i in range(4):
        faces_opposite_node = faces_opposite_node_idx[i]
        elem_gradient += grad[:, i, :] * elem_weight[:, faces_opposite_node]
    return elem_gradient
