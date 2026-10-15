import sys
import time

class Logger:
    def __init__(self, total_iters, bar_length=40):
        self.total_iters = total_iters
        self.bar_length = bar_length
        self.start_time = time.time()
        self.last_lines = 0

    def log(self, it, loss_dict, the_last20_img_loss, optimizer):
        percent = it / self.total_iters
        filled_len = int(self.bar_length * percent)
        bar = '=' * filled_len + '-' * (self.bar_length - filled_len)

        elapsed = time.time() - self.start_time
        eta = elapsed / max(it, 1) * (self.total_iters - it)

        # 构造两行日志
        line1 = (
            f"[{it:05d}/{self.total_iters}] |{bar}| "
            f"ETA: {eta:6.1f}s | "
            f"Total={loss_dict['loss_total']:.6f}, "
            f"Img={loss_dict['loss_img']:.6f}, "
            f"Last20={the_last20_img_loss:.4f}"
        )

        line2 = (
            f"WholeImg={loss_dict['loss_whole_img']:.6f}, "
            f"Lap={loss_dict['loss_lap']:.6f}, "
            f"Edge={loss_dict['loss_edge']:.6f}, "
            f"Lap3D={loss_dict['loss_laplace']:.6f}, "
            f"Dist3D={loss_dict['loss_distances']:.6f}, "
            f"LR={optimizer.param_groups[0]['lr']:.5f}   "
        )

        # 清除上一次的两行输出（用 ANSI 转义符移动光标）
        if self.last_lines:
            sys.stdout.write('\x1b[2A')  # move cursor up 2 lines
            sys.stdout.write('\x1b[2K\r')  # clear line
            sys.stdout.write('\x1b[1B\x1b[2K\r')  # go to next line and clear
            sys.stdout.write('\x1b[1A')  # move back up

        sys.stdout.write(line1 + '\n')
        sys.stdout.write(line2 + '\n')
        sys.stdout.flush()

        self.last_lines = 2

        if it == self.total_iters:
            print("")  # 输出换行避免最后一条被覆盖
