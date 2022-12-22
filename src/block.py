class Block:
    def __init__(self, block_size, current_step, dirty, address, domain_id = -1):
        self.size = block_size
        self.dirty_bit = dirty
        self.last_accessed = current_step
        self.address = address
        self.doamin_id = domain_id # for cyclone

    def is_dirty(self):
        return self.dirty_bit

    def write(self, current_step):
        self.dirty_bit = True
        self.last_accessed = current_step

    def clean(self):
        self.dirty_bit = False

    def read(self, current_step):
        self.last_accessed = current_step

