import oven.language as ol


def cos(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.cos(x_value)
    ol.store(y_value, y_ptr, idx)


def sin(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.sin(x_value)
    ol.store(y_value, y_ptr, idx)


def tan(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.tan(x_value)
    ol.store(y_value, y_ptr, idx)


def log(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.log(x_value)
    ol.store(y_value, y_ptr, idx)


def log2(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.log2(x_value)
    ol.store(y_value, y_ptr, idx)
