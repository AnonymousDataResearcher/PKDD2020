import numpy as np

from cjo.base.classes import TransactionLog
from cjo.base.stringconstants import visit, human_names_dict
from functions.general_functions import listified
from functions.tex_functions import MultipleTablesDocument


def generate_receipts(tl_source, sku_map, vids, pdf_destination, seed=None, num=None):
    tl = TransactionLog(tl_source)
    ntl = tl.to_ntl(sku_map)

    if seed is not None:
        np.random.seed(seed)
        vids = np.random.permutation(list(vids))

    vids = listified(vids, str, validation=lambda x: x in tl[visit].values)

    mtd = MultipleTablesDocument(pdf_destination)

    if num is None:
        vids = {vid: vid for vid in vids}
    else:
        vids = {vid: f'{num}{i}' for i, vid in enumerate(vids)}

    for vid in vids:
        mtd.add(ntl.subset_visit(vid).df.rename(
            columns=human_names_dict), index=False, header=True,
            caption=vid)

    mtd.produce()

    return vids
