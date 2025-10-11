import oven.language as ol


def add(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value + y_value
    ol.store(z_value, c_ptr, idx)


def mul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value * y_value
    ol.store(z_value, c_ptr, idx)


def sub(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value - y_value
    ol.store(z_value, c_ptr, idx)


def div(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value / y_value
    ol.store(z_value, c_ptr, idx)


def vadd(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = (bid * bsize + tid) * 4
    x_value = ol.vload(a_ptr, idx)
    y_value = ol.vload(b_ptr, idx)
    z_value = x_value + y_value
    ol.vstore(z_value, c_ptr, idx)


def vmul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = (bid * bsize + tid) * 4
    x_value = ol.vload(a_ptr, idx)
    y_value = ol.vload(b_ptr, idx)
    z_value = x_value * y_value
    ol.vstore(z_value, c_ptr, idx)


def vsub(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = (bid * bsize + tid) * 4
    x_value = ol.vload(a_ptr, idx)
    y_value = ol.vload(b_ptr, idx)
    z_value = x_value - y_value
    ol.vstore(z_value, c_ptr, idx)


def vdiv(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = (bid * bsize + tid) * 4
    x_value = ol.vload(a_ptr, idx)
    y_value = ol.vload(b_ptr, idx)
    z_value = x_value / y_value
    ol.vstore(z_value, c_ptr, idx)
