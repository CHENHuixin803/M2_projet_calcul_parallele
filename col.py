"""
生命游戏（MPI版本 - 列分解 / Column Decomposition）
终极修复版：修复图案初始化错位、恢复UI图案显示、NumPy矢量化满血加速
"""
import os, io, base64, time, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg, numpy as np
from mpi4py import MPI

class Grille:
    def __init__(self, rank, nbp, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.rank, self.nbp, self.global_dim = rank, nbp, dim
        nx_loc = dim[1] // nbp + (1 if rank < dim[1] % nbp else 0)
        self.x_start = nx_loc * rank + (dim[1] % nbp if rank >= dim[1] % nbp else 0)
        self.nx_loc, self.ny = nx_loc, dim[0]
        self.dimensions = (self.ny, self.nx_loc + 2) 

        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            for v in init_pattern:
                y_glob, x_glob = v[0], v[1]
                # 【修复】严格判断全局坐标是否落在本进程的列范围内，杜绝幽灵列溢出
                if self.x_start <= x_glob < self.x_start + self.nx_loc:
                    self.cells[y_glob, x_glob - self.x_start + 1] = 1
        else:
            self.cells = np.random.randint(2, size=self.dimensions, dtype=np.uint8)
        self.col_life, self.col_dead = color_life, color_dead

    def compute_next_iteration(self, globCom):
        t0 = MPI.Wtime()
        left, right = (self.rank - 1) % self.nbp, (self.rank + 1) % self.nbp
        recv_right, recv_left = np.empty(self.ny, dtype=np.uint8), np.empty(self.ny, dtype=np.uint8)
        
        globCom.Sendrecv(self.cells[:, 1].copy(), dest=left, sendtag=10, recvbuf=recv_right, source=right, recvtag=10)
        self.cells[:, self.nx_loc + 1] = recv_right
        globCom.Sendrecv(self.cells[:, self.nx_loc].copy(), dest=right, sendtag=11, recvbuf=recv_left, source=left, recvtag=11)
        self.cells[:, 0] = recv_left

        t1 = MPI.Wtime()
        # 【重点】NumPy 矢量化加速，抛弃 for 循环
        padded = np.empty((self.ny + 2, self.nx_loc + 2), dtype=np.uint8)
        padded[1:-1, :] = self.cells
        padded[0, :] = self.cells[-1, :]  # 上下环形拓扑
        padded[-1, :] = self.cells[0, :]

        neighbors = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                     padded[1:-1, :-2] +                    padded[1:-1, 2:] +
                     padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:])

        real_cells = self.cells[:, 1:self.nx_loc+1]
        new_cells = np.zeros_like(real_cells)
        new_cells[(real_cells == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        new_cells[(real_cells == 0) & (neighbors == 3)] = 1

        self.cells[:, 1:self.nx_loc+1] = new_cells
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
        
        # 【修复】传入真实的 init_pattern，不再是 None 噪点
        grid = Grille(rank, nbp, global_dim, init_pattern)
        if rank == 0: appli = App((resx, resy), global_dim, grid)
        
        all_local_nxs = globCom.gather(grid.nx_loc, root=0)
        if rank == 0:
            recvcounts = tuple([global_dim[0] * nx for nx in all_local_nxs])
            displacements = tuple([sum(recvcounts[:i]) for i in range(nbp)])
            global_cells_flat = np.empty(global_dim[0] * global_dim[1], dtype=np.uint8)
            global_cells_2d = np.empty(global_dim, dtype=np.uint8)
        else: recvcounts = displacements = global_cells_flat = global_cells_2d = None

        mustContinue_arr = np.array([1], dtype=np.uint8)
        while mustContinue_arr[0] == 1:
            t1 = time.time()
            t_comp, t_comm = grid.compute_next_iteration(globCom)
            local_real_cells = grid.cells[:, 1 : grid.nx_loc + 1].flatten()
            globCom.Gatherv(local_real_cells, [global_cells_flat, recvcounts, displacements, MPI.BYTE], root=0)
            
            t2 = time.time()
            if rank == 0:
                offset, current_x_start = 0, 0
                for r in range(nbp):
                    nx_r = all_local_nxs[r]
                    chunk = global_cells_flat[offset : offset + global_dim[0] * nx_r].reshape((global_dim[0], nx_r))
                    global_cells_2d[:, current_x_start : current_x_start + nx_r] = chunk
                    offset += global_dim[0] * nx_r
                    current_x_start += nx_r
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
        grid = Grille(rank, nbp, global_dim, init_pattern=None)
        grid.cells[:, 1 : grid.nx_loc + 1] = global_initial_cells[:, grid.x_start : grid.x_start + grid.nx_loc]

        all_local_nxs = globCom.gather(grid.nx_loc, root=0)
        if rank == 0:
            recvcounts = tuple([global_dim[0] * nx_ for nx_ in all_local_nxs])
            displacements = tuple([sum(recvcounts[:i]) for i in range(nbp)])
            global_cells_flat = np.empty(global_dim[0] * global_dim[1], dtype=np.uint8)
            global_cells_2d = np.empty(global_dim, dtype=np.uint8)
        else: recvcounts = displacements = global_cells_flat = global_cells_2d = None

        globCom.Barrier()
        t_total_start = MPI.Wtime()
        total_comp, total_comm = 0.0, 0.0

        for _ in range(n_iter):
            t_comp, t_comm = grid.compute_next_iteration(globCom)
            total_comp += t_comp
            total_comm += t_comm

        globCom.Barrier()
        t_total_end = MPI.Wtime()
        all_comp = globCom.gather(total_comp, root=0)
        all_comm = globCom.gather(total_comm, root=0)

        if dump_grid:
            local_real_cells = grid.cells[:, 1 : grid.nx_loc + 1].flatten()
            globCom.Gatherv(local_real_cells, [global_cells_flat, recvcounts, displacements, MPI.BYTE], root=0)
            if rank == 0:
                offset, current_x_start = 0, 0
                for r in range(nbp):
                    nx_r = all_local_nxs[r]
                    chunk = global_cells_flat[offset : offset + global_dim[0] * nx_r].reshape((global_dim[0], nx_r))
                    global_cells_2d[:, current_x_start : current_x_start + nx_r] = chunk
                    offset += global_dim[0] * nx_r
                    current_x_start += nx_r

        if rank == 0:
            print(f"TIME_COMPUTE={np.mean(all_comp):.6f}")
            print(f"TIME_COMM={np.mean(all_comm):.6f}")
            print(f"TIME_TOTAL={t_total_end - t_total_start:.6f}")
            if dump_grid:
                buf = io.BytesIO()
                np.save(buf, global_cells_2d)
                print(f"GRID={base64.b64encode(buf.getvalue()).decode()}")