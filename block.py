"""
生命游戏（MPI版本 - 二维块分解 Block Decomposition）
终极修复版：修复图案初始化、恢复UI图案显示、NumPy矢量化满血加速
"""
import os, io, base64, time, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg, numpy as np
from mpi4py import MPI

class Grille:
    def __init__(self, cart_comm, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.cart_comm = cart_comm
        self.rank = cart_comm.rank
        self.global_dim = dim
        topo_dims = cart_comm.dims
        coords = cart_comm.Get_coords(self.rank)
        py, px = topo_dims[0], topo_dims[1]
        cy, cx = coords[0], coords[1]
        
        base_y, rem_y = dim[0] // py, dim[0] % py
        self.ny_loc = base_y + (1 if cy < rem_y else 0)
        self.y_start = cy * base_y + (cy if cy < rem_y else rem_y)
        
        base_x, rem_x = dim[1] // px, dim[1] % px
        self.nx_loc = base_x + (1 if cx < rem_x else 0)
        self.x_start = cx * base_x + (cx if cx < rem_x else rem_x)
        
        self.dimensions = (self.ny_loc + 2, self.nx_loc + 2)
        
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            for v in init_pattern:
                if (self.y_start <= v[0] < self.y_start + self.ny_loc) and (self.x_start <= v[1] < self.x_start + self.nx_loc):
                    self.cells[v[0] - self.y_start + 1, v[1] - self.x_start + 1] = 1
        else:
            self.cells = np.random.randint(2, size=self.dimensions, dtype=np.uint8)
        self.col_life, self.col_dead = color_life, color_dead

    def compute_next_iteration(self):
        t0 = MPI.Wtime()
        top, bot = self.cart_comm.Shift(0, 1) 
        left, right = self.cart_comm.Shift(1, 1) 
        
        self.cart_comm.Sendrecv(self.cells[1, 1:self.nx_loc+1].copy(), dest=top, sendtag=10, recvbuf=self.cells[self.ny_loc+1, 1:self.nx_loc+1], source=bot, recvtag=10)
        self.cart_comm.Sendrecv(self.cells[self.ny_loc, 1:self.nx_loc+1].copy(), dest=bot, sendtag=11, recvbuf=self.cells[0, 1:self.nx_loc+1], source=top, recvtag=11)
                                
        recv_right_col, recv_left_col = np.empty(self.ny_loc + 2, dtype=np.uint8), np.empty(self.ny_loc + 2, dtype=np.uint8)
        self.cart_comm.Sendrecv(self.cells[:, 1].copy(), dest=left, sendtag=12, recvbuf=recv_right_col, source=right, recvtag=12)
        self.cells[:, self.nx_loc + 1] = recv_right_col 
        self.cart_comm.Sendrecv(self.cells[:, self.nx_loc].copy(), dest=right, sendtag=13, recvbuf=recv_left_col, source=left, recvtag=13)
        self.cells[:, 0] = recv_left_col 

        t1 = MPI.Wtime()
        # NumPy 矢量化加速，利用四周边界极其优雅！
        neighbors = (self.cells[:-2, :-2] + self.cells[:-2, 1:-1] + self.cells[:-2, 2:] +
                     self.cells[1:-1, :-2] +                        self.cells[1:-1, 2:] +
                     self.cells[2:, :-2]  + self.cells[2:, 1:-1]  + self.cells[2:, 2:])

        real_cells = self.cells[1:self.ny_loc+1, 1:self.nx_loc+1]
        new_cells = np.zeros_like(real_cells)
        new_cells[(real_cells == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        new_cells[(real_cells == 0) & (neighbors == 3)] = 1

        self.cells[1:self.ny_loc+1, 1:self.nx_loc+1] = new_cells
        t2 = MPI.Wtime()
        return (t2 - t1), (t1 - t0)

class App:
    def __init__(self, geometry, global_dim, grid):
        self.grid, self.global_dim = grid, global_dim
        self.size_x, self.size_y = geometry[1] // global_dim[1], geometry[0] // global_dim[0]
        self.draw_color = pg.Color('lightgrey') if (self.size_x > 4 and self.size_y > 4) else None
        self.width, self.height = global_dim[1] * self.size_x, global_dim[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
    def draw(self, global_cells):
        [self.screen.fill(self.grid.col_dead if global_cells[i,j]==0 else self.grid.col_life, (self.size_x*j, self.height - self.size_y*(i + 1), self.size_x, self.size_y)) for i in range(self.global_dim[0]) for j in range(self.global_dim[1])]
        if self.draw_color is not None:
            [pg.draw.line(self.screen, self.draw_color, (0, i*self.size_y), (self.width, i*self.size_y)) for i in range(self.global_dim[0])]
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x, 0), (j*self.size_x, self.height)) for j in range(self.global_dim[1])]
        pg.display.update()

if __name__ == '__main__':
    globCom = MPI.COMM_WORLD.Dup() 
    nbp, rank = globCom.size, globCom.rank         
    dims = MPI.Compute_dims(nbp, 2)
    cart_comm = globCom.Create_cart(dims=dims, periods=(True, True), reorder=False)

    benchmark_mode = len(sys.argv) > 1 and sys.argv[1] == '--benchmark'
    if rank == 0 and not benchmark_mode: pg.init()
    
    dico_patterns = {
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)])
    }

    if not benchmark_mode:
        choice = sys.argv[1] if len(sys.argv) > 1 else 'glider'
        resx = int(sys.argv[2]) if len(sys.argv) > 3 else 800
        resy = int(sys.argv[3]) if len(sys.argv) > 3 else 800
        
        if choice not in dico_patterns:
            if rank == 0: print("图案未知")
            MPI.Finalize()
            sys.exit(1)
            
        global_dim, init_pattern = dico_patterns[choice]

        grid = Grille(cart_comm, global_dim, init_pattern)
        if rank == 0: appli = App((resx, resy), global_dim, grid)
        
        dims_list = globCom.gather((grid.ny_loc, grid.nx_loc), root=0)
        if rank == 0:
            recvcounts = tuple([d[0] * d[1] for d in dims_list])
            displacements = tuple([sum(recvcounts[:i]) for i in range(nbp)])
            global_cells_flat = np.empty(global_dim[0] * global_dim[1], dtype=np.uint8)
            global_cells_2d = np.empty(global_dim, dtype=np.uint8)
        else: recvcounts = displacements = global_cells_flat = global_cells_2d = None

        mustContinue_arr = np.array([1], dtype=np.uint8)
        while mustContinue_arr[0] == 1:
            t1 = time.time()
            grid.compute_next_iteration()
            local_real_cells = grid.cells[1 : grid.ny_loc+1, 1 : grid.nx_loc+1].flatten()
            globCom.Gatherv(local_real_cells, [global_cells_flat, recvcounts, displacements, MPI.BYTE], root=0)
            
            t2 = time.time()
            if rank == 0:
                offset = 0
                for r in range(nbp):
                    ny_r, nx_r = dims_list[r]
                    c_y, c_x = cart_comm.Get_coords(r)
                    y_st = c_y * (global_dim[0]//dims[0]) + (c_y if c_y < global_dim[0]%dims[0] else global_dim[0]%dims[0])
                    x_st = c_x * (global_dim[1]//dims[1]) + (c_x if c_x < global_dim[1]%dims[1] else global_dim[1]%dims[1])
                    chunk = global_cells_flat[offset : offset + ny_r*nx_r].reshape((ny_r, nx_r))
                    global_cells_2d[y_st : y_st+ny_r, x_st : x_st+nx_r] = chunk
                    offset += ny_r*nx_r
                appli.draw(global_cells_2d)
                for event in pg.event.get():
                    if event.type == pg.QUIT: mustContinue_arr[0] = 0
                print(f"MPI 计算与通信: {t2-t1:2.2e} s | 渲染: {time.time()-t2:2.2e} s\r", end='', flush=True)
            globCom.Bcast(mustContinue_arr, root=0)
        if rank == 0: pg.quit()

    else:
        # Benchmark 模式
        ny, nx, n_iter = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        seed = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '--dump-grid' else 42
        dump_grid = '--dump-grid' in sys.argv
        global_dim = (ny, nx)

        global_initial_cells = np.random.RandomState(seed).randint(0, 2, size=global_dim, dtype=np.uint8)
        grid = Grille(cart_comm, global_dim, init_pattern=None)
        grid.cells[1:grid.ny_loc+1, 1:grid.nx_loc+1] = global_initial_cells[grid.y_start:grid.y_start+grid.ny_loc, grid.x_start:grid.x_start+grid.nx_loc]

        dims_list = globCom.gather((grid.ny_loc, grid.nx_loc), root=0)
        if rank == 0:
            recvcounts = tuple([d[0] * d[1] for d in dims_list])
            displacements = tuple([sum(recvcounts[:i]) for i in range(nbp)])
            global_cells_flat = np.empty(global_dim[0] * global_dim[1], dtype=np.uint8)
            global_cells_2d = np.empty(global_dim, dtype=np.uint8)
        else: recvcounts = displacements = global_cells_flat = global_cells_2d = None

        globCom.Barrier()
        t_total_start = MPI.Wtime()
        total_comp, total_comm = 0.0, 0.0

        for _ in range(n_iter):
            t_comp, t_comm = grid.compute_next_iteration()
            total_comp += t_comp
            total_comm += t_comm

        globCom.Barrier()
        t_total_end = MPI.Wtime()
        all_comp = globCom.gather(total_comp, root=0)
        all_comm = globCom.gather(total_comm, root=0)

        if dump_grid:
            local_real_cells = grid.cells[1:grid.ny_loc+1, 1:grid.nx_loc+1].flatten()
            globCom.Gatherv(local_real_cells, [global_cells_flat, recvcounts, displacements, MPI.BYTE], root=0)
            if rank == 0:
                offset = 0
                for r in range(nbp):
                    ny_r, nx_r = dims_list[r]
                    c_y, c_x = cart_comm.Get_coords(r)
                    y_st = c_y * (global_dim[0]//dims[0]) + (c_y if c_y < global_dim[0]%dims[0] else global_dim[0]%dims[0])
                    x_st = c_x * (global_dim[1]//dims[1]) + (c_x if c_x < global_dim[1]%dims[1] else global_dim[1]%dims[1])
                    chunk = global_cells_flat[offset : offset + ny_r*nx_r].reshape((ny_r, nx_r))
                    global_cells_2d[y_st : y_st+ny_r, x_st : x_st+nx_r] = chunk
                    offset += ny_r*nx_r

        if rank == 0:
            print(f"TIME_COMPUTE={np.mean(all_comp):.6f}")
            print(f"TIME_COMM={np.mean(all_comm):.6f}")
            print(f"TIME_TOTAL={t_total_end - t_total_start:.6f}")
            if dump_grid:
                buf = io.BytesIO()
                np.save(buf, global_cells_2d)
                print(f"GRID={base64.b64encode(buf.getvalue()).decode()}")