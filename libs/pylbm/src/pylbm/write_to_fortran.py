"""
Utilities for writing and modifying Fortran source files.
"""

import pathlib

from . import LBM_PATH


def add_module_to_main(case_name: str) -> None:
    """
    Add use statement for m_{case_name} to main.F90.

    Args:
        case_name: Name of the case (e.g., "runcase")
    """
    main_f90_path = LBM_PATH / "src" / "main.F90"  # type: ignore[operator]

    if not main_f90_path.exists():
        raise FileNotFoundError(f"main.F90 not found at {main_f90_path}")

    # Read main.F90
    content = main_f90_path.read_text()
    lines = content.splitlines()

    # Find the use statements section (lines 5-51)
    # Insert the new use statement in alphabetical order
    module_name = f"m_{case_name}"
    use_statement = f"   use {module_name}"

    # Find insertion point (alphabetically after existing use statements)
    insert_idx = None
    for i, line in enumerate(lines):
        # Skip the initial program and #ifdef lines
        if i < 4:
            continue
        # Stop at implicit none or other non-use statements
        if line.strip().startswith("use, intrinsic") or line.strip().startswith(
            "implicit"
        ):
            insert_idx = i
            break
        # Check if we should insert before this line (alphabetically)
        if line.strip().startswith("use "):
            existing_module = line.strip().replace("use ", "").strip()
            if module_name < existing_module:
                insert_idx = i
                break

    # If we didn't find an insertion point, insert before implicit none
    if insert_idx is None:
        # Find implicit none line
        for i, line in enumerate(lines):
            if line.strip().startswith("implicit"):
                insert_idx = i
                break

    if insert_idx is None:
        raise ValueError("Could not find insertion point in main.F90")

    # Check if the module is already included (check in use statements section)
    for line in lines[4:insert_idx]:
        if line.strip() == use_statement.strip():
            return  # Already added

    # Insert the new use statement
    lines.insert(insert_idx, use_statement)

    # Write back to file
    main_f90_path.write_text("\n".join(lines) + "\n")


def add_case_dimensions_to_mod_dimensions(
    case_name: str, nx: int, ny: int, nz: int
) -> None:
    """
    Add case dimensions to mod_dimensions.F90 and comment out all other cases.
    If the case already exists, modify it instead of adding a duplicate.

    Args:
        case_name: Name of the case (e.g., "runcase")
        nx: Grid resolution in x-direction
        ny: Grid resolution in y-direction
        nz: Grid resolution in z-direction
    """
    mod_dimensions_path = LBM_PATH / "src" / "mod_dimensions.F90"  # type: ignore[operator]

    if not mod_dimensions_path.exists():
        raise FileNotFoundError(
            f"mod_dimensions.F90 not found at {mod_dimensions_path}"
        )

    # Read mod_dimensions.F90
    content = mod_dimensions_path.read_text()
    lines = content.splitlines()

    new_lines = []
    i = 0
    case_found = False
    case_comment_pattern = f"!{case_name}"

    # Process all lines
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if we've reached the end module
        if stripped.startswith("end module"):
            # If case wasn't found, add it before end module
            if not case_found:
                new_lines.append("")
                new_lines.append(f"!{case_name}")
                new_lines.append(
                    f"  integer, parameter :: nx = {nx:<10} ! resolution x-dir (east)"
                )
                new_lines.append(
                    f"  integer, parameter :: ny = {ny:<10} ! resolution y-dir (north)"
                )
                new_lines.append(
                    f"  integer, parameter :: nz = {nz:<10} ! resolution z-dir (up)"
                )
                new_lines.append("")
            new_lines.append(line)
            break

        # Check if this is the case comment line we're looking for
        if stripped == case_comment_pattern:
            case_found = True
            new_lines.append(line)  # Keep the comment line
            i += 1

            # Skip empty line if present
            if i < len(lines) and lines[i].strip() == "":
                new_lines.append(lines[i])
                i += 1

            # Replace the integer parameter lines with new values (uncommented)
            param_count = 0
            while i < len(lines) and "integer, parameter" in lines[i]:
                param_line = lines[i]
                param_count += 1

                if param_count == 1:
                    # nx parameter
                    new_lines.append(
                        f"  integer, parameter :: nx = {nx:<10} ! resolution x-dir (east)"
                    )
                elif param_count == 2:
                    # ny parameter
                    new_lines.append(
                        f"  integer, parameter :: ny = {ny:<10} ! resolution y-dir (north)"
                    )
                elif param_count == 3:
                    # nz parameter
                    new_lines.append(
                        f"  integer, parameter :: nz = {nz:<10} ! resolution z-dir (up)"
                    )
                else:
                    # Extra parameters - skip them (shouldn't happen normally)
                    pass

                i += 1

            # Skip empty line after case if present
            if i < len(lines) and lines[i].strip() == "":
                new_lines.append(lines[i])
                i += 1
            continue

        # Check if this is an uncommented integer parameter line (active case from another case)
        if stripped.startswith("integer, parameter") and not stripped.startswith("!"):
            # This is an active case definition from a different case - comment it out
            new_lines.append("!" + line)
            i += 1
            # Comment out subsequent integer parameter lines in this case
            while i < len(lines) and "integer, parameter" in lines[i]:
                param_line = lines[i]
                if not param_line.strip().startswith("!"):
                    new_lines.append("!" + param_line)
                else:
                    new_lines.append(param_line)
                i += 1
            # Skip empty line after case if present
            if i < len(lines) and lines[i].strip() == "":
                new_lines.append(lines[i])
                i += 1
            continue

        # Regular line - keep as is
        new_lines.append(line)
        i += 1

    # Write back to file
    mod_dimensions_path.write_text("\n".join(new_lines) + "\n")
