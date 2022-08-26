import numpy as np

class ConnectionProperty():
    def __init__(self, connected_node, this_node, ihc, cl1, cl2, hwva, angldegx):
        self.connected_node = connected_node
        self.this_node = this_node
        self.ihc = ihc
        self.cl1 = cl1
        self.cl2 = cl2
        self.hwva = hwva
        self.angldegx = angldegx


class ConnectionPropertyList():
    def __init__(self):
        self.number_of_connections = None
        self.connection_dict = {}
    
    def new_connection(self, connected_node, this_node, ihc, cl1, cl2, hwva, angldegx):
        connection = ConnectionProperty(connected_node, this_node, ihc, cl1, cl2, hwva, angldegx)
        if self.number_of_connections is None:
            connection_number = 0
            self.number_of_connections = 0
        else:
            connection_number = self.number_of_connections
        self.connection_dict[connection_number] = connection
        self.number_of_connections += 1
        return connection_number


class Dis2Disu():

    def __init__(self, delr, delc, top, botm, staggered):
        
        assert isinstance(delr, np.ndarray)
        assert isinstance(delc, np.ndarray)
        assert isinstance(top, np.ndarray)
        assert isinstance(botm, np.ndarray)

        assert delr.ndim == 1, f"delr must have ndim = 1 but found {delr.ndim}"
        assert delc.ndim == 1, f"delc must have ndim = 1 but found {delc.ndim}"
        assert top.ndim == 2, f"top must have ndim = 2 but found {top.ndim}"
        assert botm.ndim == 3, f"botm must have ndim = 3 but found {botm.ndim}"

        self.nlay, self.nrow, self.ncol = botm.shape
        self._nodes = self.nlay * self.nrow * self.ncol

        self.delr = delr
        self.delc = delc
        self.top = top
        self.botm = botm
        self.staggered = staggered

        self.connection_list = ConnectionPropertyList()
        self._initialize()
        self._build_connectivity()

        return

    def _initialize(self):

        # unknown at this point
        self._nja = None
        self._nvert = None
        self._iac = np.zeros(self._nodes, dtype=int)
        self._ja = None
        self._ihc = None
        self._cl12 = None
        self._hwva = None
        self._angldegx = None
        self._vertices = None
        self._cell2d = None

        # can be calculated
        self._top = self.top.flatten()
        top = []
        for k in range(self.nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    tp, _ = self.get_cell_topbot(k, i, j)
                    top.append(tp)
        self._top = np.array(top, dtype=float)

        self._bot = self.botm.flatten()
        area = np.empty((self.nlay, self.nrow, self.ncol), dtype=float)
        for i in range(self.nrow):
            for j in range(self.ncol):
                area[:, i, j] = self.delr[j] * self.delc[i]
        self._area = np.array(area, dtype=float).flatten()

        return

    def _build_connectivity(self):
        list_of_connections = []
        kk, ii, jj = np.where(np.ones((self.nlay, self.nrow, self.ncol)) == 1)
        for icell, (k, i, j) in enumerate(zip(kk, ii, jj)):
            connections = self.get_cell_connections(k, i, j)
            self._iac[icell] = len(connections) + 1
            list_of_connections.append(connections)
        self.list_of_connections = list_of_connections

        self._nja = self._iac.sum()
        self._allocate_connection_arrays(self.nja)
        self._fill_connection_arrays()
        self._fill_vertices()
        self._fill_cell2d()

        return

    def _allocate_connection_arrays(self, nja):
        self._ja = np.zeros(nja, dtype=int)
        self._ihc = np.zeros(nja, dtype=int)
        self._cl12 = np.zeros(nja, dtype=float)
        self._hwva = np.zeros(nja, dtype=float)
        self._angldegx = np.zeros(nja, dtype=float)
        return

    def _fill_connection_arrays(self):
        ipos = 0
        for icell, cell_connections in enumerate(self.list_of_connections):
            layer, row, column = self.get_kij(icell)
            self._ja[ipos] = icell
            # store layer number in ihc diagonal for flopy plotting
            self._ihc[ipos] = layer
            self._cl12[ipos] = icell
            self._hwva[ipos] = icell
            self._angldegx[ipos] = icell
            ipos += 1
            cc = cell_connections.copy()
            cc.sort()
            for nn, cn in cc:
                cp = self.connection_list.connection_dict[cn]
                assert icell == cp.this_node
                self._ja[ipos] = cp.connected_node
                self._ihc[ipos] = cp.ihc
                self._cl12[ipos] = cp.cl1
                self._hwva[ipos] = cp.hwva
                self._angldegx[ipos] = cp.angldegx
                ipos += 1
        return

    def _fill_vertices(self):
        # number vertices from left to right and from back to front
        xe = np.array([0] + list(self.delr)).cumsum()
        y0 = self.delc.sum()
        ye = [y0] + list(self.delc.sum() - self.delc.cumsum())
        ye = np.array(ye)
        xv, yv = np.meshgrid(xe, ye)
        xv = xv.flatten()
        yv = yv.flatten()
        self._nvert = xv.shape[0]
        vertices = []
        for ivert in range(self._nvert):
            vertices.append([ivert, xv[ivert], yv[ivert]])
        self._vertices = vertices
        self._nvert = len(vertices)
        return

    def _fill_cell2d(self):
        cell2d = []
        for k in range(self.nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    nn = self.get_nodenumber(k, i, j)
                    iv1 = j + i * (self.ncol + 1)  # upper left
                    iv2 = iv1 + 1
                    iv3 = iv2 + self.ncol + 1
                    iv4 = iv3 - 1
                    _, xul, yul = self._vertices[iv1]
                    _, xlr, ylr = self._vertices[iv3]
                    xc = 0.5 * (xul + xlr)
                    yc = 0.5 * (yul + ylr)
                    c2d = [nn, xc, yc, 5, iv1, iv2, iv3, iv4, iv1]
                    cell2d.append(c2d)
        self._cell2d = cell2d

    def get_gridprops_disu6(self):
        gridprops = {}
        gridprops["nodes"] = self._nodes
        gridprops["top"] = self._top
        gridprops["bot"] = self._bot
        gridprops["area"] = self._area
        gridprops["iac"] = self._iac
        gridprops["nja"] = self._nja
        gridprops["ja"] = self._ja
        gridprops["cl12"] = self._cl12
        gridprops["ihc"] = self._ihc
        gridprops["hwva"] = self._hwva
        gridprops["angldegx"] = self._angldegx
        gridprops["nvert"] = self._nvert
        gridprops["vertices"] = self._vertices
        gridprops["cell2d"] = self._cell2d
        return gridprops

    @property
    def iac(self):
        return self._iac
        
    @property
    def nja(self):
        return self._nja
        
    def get_cell_connections(self, k, i, j):
        # return list of connections as [(nn, connection_number), ...]
        nn = self.get_nodenumber(k, i, j)
        this_node = nn
        connected_nodes = []
        if k > 0: # top
            nn = self.get_nodenumber(k - 1, i, j)
            ihc = 0
            cl1 = 0.5 * (self._top[this_node] - self._bot[this_node])
            cl2 = 0.5 * (self._top[nn] - self._bot[nn])
            hwva = self._area[nn]
            angldegx = 0.
            connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
            connected_nodes.append((nn, connection_number))
        if i > 0: # back
            angldegx = 90.
            if self.staggered:
                connections = self.get_connected_staggered((k, i, j), (k, i - 1, j), angldegx)
                connected_nodes.extend(connections)
            else:
                nn = self.get_nodenumber(k, i - 1, j)
                ihc = 1
                cl1 = 0.5 * self.delc[i]
                cl2 = 0.5 * self.delc[i - 1]
                hwva = self.delr[j]
                connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
                connected_nodes.append((nn, connection_number))
        if j > 0: # left
            angldegx = 180.
            if self.staggered:
                connections = self.get_connected_staggered((k, i, j), (k, i, j - 1), angldegx)
                connected_nodes.extend(connections)
            else:
                nn = self.get_nodenumber(k, i, j - 1)
                ihc = 1
                cl1 = 0.5 * self.delr[j]
                cl2 = 0.5 * self.delr[j - 1]
                hwva = self.delc[i]
                connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
                connected_nodes.append((nn, connection_number))
        if j < self.ncol - 1: # right
            angldegx = 0.
            if self.staggered:
                connections = self.get_connected_staggered((k, i, j), (k, i, j + 1), angldegx)
                connected_nodes.extend(connections)
            else:
                nn = self.get_nodenumber(k, i, j + 1)
                ihc = 1
                cl1 = 0.5 * self.delr[j]
                cl2 = 0.5 * self.delr[j + 1]
                hwva = self.delc[i]
                connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
                connected_nodes.append((nn, connection_number))
        if i < self.nrow - 1: # front
            angldegx = 270.
            if self.staggered:
                connections = self.get_connected_staggered((k, i, j), (k, i + 1, j), angldegx)
                connected_nodes.extend(connections)
            else:
                nn = self.get_nodenumber(k, i + 1, j)
                ihc = 1
                cl1 = 0.5 * self.delc[i]
                cl2 = 0.5 * self.delc[i + 1]
                hwva = self.delr[j]
                connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
                connected_nodes.append((nn, connection_number))
        if k < self.nlay - 1: # bottom
            nn = self.get_nodenumber(k + 1, i, j)
            ihc = 0
            cl1 = 0.5 * (self._top[this_node] - self._bot[this_node])
            cl2 = 0.5 * (self._top[nn] - self._bot[nn])
            hwva = self._area[nn]
            angldegx = 0.
            connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
            connected_nodes.append((nn, connection_number))
        return connected_nodes

    def get_cell_topbot(self, k, i, j):
        if k == 0:
            top = self.top[i, j]
        else:
            top = self.botm[k - 1, i, j]
        bot = self.botm[k, i, j]
        return top, bot

    def get_connected_staggered(self, this_cell, connected_cell, angldegx):
        connected_cells = []
        k, i, j = this_cell
        this_node = self.get_nodenumber(k, i, j)
        this_top, this_bot = self.get_cell_topbot(k, i, j)
        k1, i1, j1 = connected_cell
        # look down
        for ksearch in range(self.nlay):
            top, bot = self.get_cell_topbot(ksearch, i1, j1)
            dz = min(this_top, top) - max(this_bot, bot)
            if dz > 0:
                # there is overlap so add a connection
                nn = self.get_nodenumber(ksearch, i1, j1)
                ihc = 2
                if angldegx == 0: # right
                    cl1 = 0.5 * self.delr[j]
                    cl2 = 0.5 * self.delr[j + 1]
                    #hwva = 0.5 * self.delc[i]
                    hwva = self.delc[i]
                elif angldegx == 90.: # back
                    cl1 = 0.5 * self.delc[i]
                    cl2 = 0.5 * self.delc[i - 1]
                    #hwva = 0.5 * self.delr[j]
                    hwva = self.delr[j]
                elif angldegx == 180: # left
                    cl1 = 0.5 * self.delr[j]
                    cl2 = 0.5 * self.delr[j - 1]
                    #hwva = 0.5 * self.delc[i]
                    hwva = self.delc[i]
                elif angldegx == 270.: # front
                    cl1 = 0.5 * self.delc[i]
                    cl2 = 0.5 * self.delc[i + 1]
                    #hwva = 0.5 * self.delr[j]
                    hwva = self.delr[j]
                connection_number = self.connection_list.new_connection(nn, this_node, ihc, cl1, cl2, hwva, angldegx)
                connected_cells.append((nn, connection_number))
        return connected_cells

    def get_nodenumber(self, k, i, j):
        cells_per_layer = self.nrow * self.ncol
        return cells_per_layer * k + self.ncol * i + j

    def get_kij(self, nn):
      ilay = nn / (self.ncol * self.nrow)
      ij = nn - ilay * self.ncol * self.nrow
      irow = ij / self.ncol
      icol = ij - irow * self.ncol
      return ilay, irow, icol
