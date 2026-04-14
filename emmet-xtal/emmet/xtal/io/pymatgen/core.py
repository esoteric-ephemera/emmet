"""Define core adaptors between emmet-xtal and pymatgen."""

from emmet.xtal.core.atom import Atom, AtomSymbol
from emmet.xtal.core.molecule import Molecule

from pymatgen.core import (
    Element,
    Species,
    Lattice,
    Molecule as PmgMolecule,
    Structure,
)

def atom_symbol_to_element(atom_symbol : AtomSymbol) -> Element:
    return Element(atom_symbol.name)

def specieslike_to_atom(species_like : Element | Species) -> Atom:
    kwargs = {}
    if spin := getattr(species, "spin", None):
        kwargs["spin"] = spin
    return Atom.from_str(str(species), **kwargs)

def atom_to_specieslike(atom : Atom) -> Element | Species:
    if atom.charge or atom.spin:
        return Species(
            atom.name,
            oxidation_state=atom.charge,
            spin=atom.spin,
        )
    return Element(atom.name)

def pmg_molecule_to_molecule(mol : PmgMolecule) -> Molecule:

    props = {
        k : [getattr(site.specie,pmg_k,None) for site in mol]
        for k, pmg_k in {
            "charges": "oxi_state",
            "spins": "spin"
        }.items()
    }
    for k, vals in props.items():
        if any(chg is not None for chg in charges):
            props[k] = [v or 0. for v in vals]
        else:
            props[k] = None

    spins = [getattr(site.specie,"spin",None) for site in mol]

    return Molecule(
        [site.specie.Z for site in mol],
        mol.cart_coords.tolist(),
        **props,
        **{
            k : mol.site_properties.get(pmg_k,None)
            for k, pmg_k in {
                "velocities": "velocities",
                "degrees_of_freedom": "selective_dynamics",
            }
        }
    )