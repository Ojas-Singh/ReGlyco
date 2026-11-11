"""Ideal bond lengths and angles of the sugars GlycoFlow fits, from the CCP4 monomer library
(https://github.com/MonomerLibrary/monomers), in GLYCAM residue codes and atom names.

    python make_ideal_geometry.py <monomer library checkout or directory of cif files> <residue_library.json>

Writes ideal_geometry.json next to this script. Needs gemmi.
"""
import json
import pathlib
import sys

import gemmi

# GLYCAM residue code -> CCD component
COMPONENTS = {
    "YB": "NAG", "YA": "NDG", "MB": "BMA", "MA": "MAN", "LB": "GAL", "LA": "GLA", "GB": "BGC", "GA": "GLC",
    "VB": "NGA", "VA": "A2G", "fA": "FUC", "fB": "FUL", "XB": "XYP", "XA": "XYS", "SA": "SIA", "SB": "SLB",
}
# CCD atom name -> GLYCAM atom name where they differ
RENAME = {
    **{c: {"C7": "C2N", "C8": "CME", "O7": "O2N"} for c in ("NAG", "NDG", "NGA", "A2G")},
    **{c: {"C10": "C5N", "C11": "CME", "O10": "O5N"} for c in ("SIA", "SLB")},
}
# the hydroxyl the glycosidic bond replaces
ANOMERIC_O = {"SIA": "O2", "SLB": "O2"}
# (anomeric carbon number)-(position) -> monomer library link
LINKS = {"1-2": "BETA1-2", "1-3": "BETA1-3", "1-4": "BETA1-4", "1-6": "BETA1-6", "2-3": "ALPHA2-3", "2-6": "ALPHA2-6", "2-8": "ALPHA2-8"}
# protein residue -> monomer library link
ANCHORS = {"ASN": "pyr-ASN", "SER": "pyr-SER", "THR": "pyr-THR", "TRP": "TRP-MAN"}


def component(root: pathlib.Path, ccd: str):
    for path in (root / ccd[0].lower() / f"{ccd}.cif", root / f"{ccd}.cif"):
        if path.exists():
            return gemmi.make_chemcomp_from_block(gemmi.cif.read(str(path))[-1])
    raise FileNotFoundError(ccd)


def residue(root, ccd, template):
    cc = component(root, ccd)
    # the polar hydrogens of the template: GLYCAM H2N, H3O for the library's HN2, HO3
    polar = {a[0] + a[2:] + a[1]: a for a, e in zip(template["atoms"], template["elements"]) if e == "H" and len(a) == 3}
    skip = {a.id for a in cc.atoms if a.el.name == "H" and a.id not in polar} | {ANOMERIC_O.get(ccd, "O1")}
    name = lambda n: polar.get(n) or RENAME.get(ccd, {}).get(n, n)
    # a hydrogen at its nucleus, as the templates have it
    length = lambda b: b.value_nucleus if "H" in (b.id1.atom[0], b.id2.atom[0]) else b.value
    bonds = [[name(b.id1.atom), name(b.id2.atom), round(length(b), 3)] for b in cc.rt.bonds if not {b.id1.atom, b.id2.atom} & skip]
    angles = [
        [name(a.id1.atom), name(a.id2.atom), name(a.id3.atom), round(a.value, 2)]
        for a in cc.rt.angles
        if not {a.id1.atom, a.id2.atom, a.id3.atom} & skip
    ]
    return {"ccd": ccd, "bonds": bonds, "angles": angles}


def link(doc, name):
    """Bond and angles of a link: the bond, the angle at the atom the anomeric carbon binds and the
    angles at the anomeric carbon by the name of its other neighbour."""
    block = doc.find_block(f"link_{name}")
    row = block.find("_chem_link_bond.", ["atom_1_comp_id", "atom_id_1", "atom_2_comp_id", "atom_id_2", "value_dist"])[0]
    ends = [(row[0], row[1]), (row[2], row[3])]
    # the anomeric carbon: the sugar's end of the bond (the second component of TRP-MAN)
    carbon = ends[1] if name == "TRP-MAN" else next(e for e in ends if e[1] in ("C1", "C2"))
    out = {"bond": float(row[4]), "anomeric": {}}
    columns = ["atom_1_comp_id", "atom_id_1", "atom_2_comp_id", "atom_id_2", "atom_3_comp_id", "atom_id_3", "value_angle"]
    for a in block.find("_chem_link_angle.", columns):
        first, centre, last = (a[0], a[1]), (a[2], a[3]), (a[4], a[5])
        if first[1].startswith("H") or last[1].startswith("H"):
            continue
        if centre == carbon:
            out["anomeric"][next(e for e in (first, last) if e not in ends)[1]] = float(a[6])
        elif centre in ends:
            # of two (Trp: CG-CD1-C1 and NE1-CD1-C1), the first
            out.setdefault("angle", float(a[6]))
    return out


def main():
    root, library = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
    doc = gemmi.cif.read(str(next(p for p in (root / "list" / "mon_lib_list.cif", root / "mon_lib_list.cif") if p.exists())))
    out = {
        "source": "CCP4 monomer library, https://github.com/MonomerLibrary/monomers",
        "residues": {code: residue(root, ccd, library["templates"][code]) for code, ccd in COMPONENTS.items()},
        "links": {key: link(doc, name) for key, name in LINKS.items()},
        "anchors": {res: link(doc, name) for res, name in ANCHORS.items()},
    }
    # every bond of the library's templates has a value, and every angle between two of them
    for code, entry in out["residues"].items():
        template = library["templates"][code]
        known = {frozenset(b[:2]) for b in entry["bonds"]}
        heavy = lambda i: True
        for a, b in template["bonds"]:
            pair = frozenset((template["atoms"][a], template["atoms"][b]))
            assert not (heavy(a) and heavy(b)) or pair in known, (code, sorted(pair))
        centres = {}
        for a, b in template["bonds"]:
            if heavy(a) and heavy(b):
                centres.setdefault(a, []).append(b)
                centres.setdefault(b, []).append(a)
        angles = {(frozenset((x[0], x[2])), x[1]) for x in entry["angles"]}
        for c, nb in centres.items():
            for i in range(len(nb)):
                for j in range(i):
                    key = (frozenset((template["atoms"][nb[i]], template["atoms"][nb[j]])), template["atoms"][c])
                    assert key in angles, (code, key)
    path = pathlib.Path(__file__).with_name("ideal_geometry.json")
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(path, path.stat().st_size, "bytes")
    print(json.dumps(out["links"], indent=1)[:900])
    print(json.dumps(out["anchors"], indent=1))


main()
