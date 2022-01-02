import math


class Patch:
    def __init__(self, id, x, y, x_end, y_end, column, row):
        self.id = id
        self.column = column
        self.row = row
        self.x = x
        self.x_end = x_end
        self.y = y
        self.y_end = y_end

        self.patch_size_x = self.x_end-self.x
        self.patch_size_y = self.y_end-self.y
        self.done = False


class Zlicer:
    def __init__(self, img, patch_size, margin):
        self.patch_size = patch_size
        self.margin = margin

        self.img_rows, self.img_cols, _ = img.shape

        self.patch_count = 0
        self.columns = 0
        self.rows = 0

        self.patches = list()

        self.bottom_edge_sample = None
        self.right_edge_sample = None

        self.initialize()

        self.bottom_edge_sample = self.get_bottom_sample()
        self.right_edge_sample = self.get_right_sample()

    def initialize(self):
        self.patch_count = 0
        self.rows = 0
        self.columns = 0

        self.columns = math.floor(self.img_cols / self.patch_size)
        self.rows = math.floor(self.img_rows / self.patch_size)

        current_row = 0
        i = 0
        while i < self.img_rows:
            i_end = min(i + self.patch_size, self.img_rows)
            j = 0
            current_col = 0
            while j < self.img_cols:
                j_end = min(j + self.patch_size, self.img_cols)

                self.patches.append(Patch(id=self.patch_count, x=j, y=i, x_end=j_end, y_end=i_end, column=current_col, row=current_row))

                self.patch_count += 1
                current_col += 1
                j = j_end
            current_row += 1
            i = i_end

    def get_bottom_sample(self):
        return self.patches[len(self.patches)-2]

    def get_right_sample(self):
        for patch in self.patches:
            if patch.column == self.columns:
                return patch