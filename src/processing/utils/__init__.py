#!/usr/bin/env python3
"""
Processing utilities — the consumers of the layout and PDF seams.

Ported from V5 module by module (see the decomposition table in plan.md).
Imports are left to the caller rather than re-exported here, so that pulling in
one utility does not drag in the dependencies of all the others.
"""
