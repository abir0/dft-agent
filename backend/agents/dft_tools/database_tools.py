"""
Database Interface Tools

Tools for managing DFT calculations database and storing results.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


@tool
def create_calculations_database(
    db_path: str = "dft_calculations.db", overwrite: bool = False
) -> str:
    """Create database for storing DFT calculation results.

    Args:
        db_path: Path to database file
        overwrite: Whether to overwrite existing database

    Returns:
        Database creation status
    """
    try:
        db_file = Path(db_path)

        if db_file.exists() and not overwrite:
            return (
                f"Database already exists at {db_path}. Use overwrite=True to recreate."
            )

        # Create database and tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Calculations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                calculation_type TEXT NOT NULL,
                structure_file TEXT,
                input_parameters TEXT,
                status TEXT DEFAULT 'pending',
                created_date TEXT,
                completed_date TEXT,
                total_energy REAL,
                convergence_achieved BOOLEAN,
                notes TEXT
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calculation_id INTEGER,
                property_name TEXT,
                property_value REAL,
                units TEXT,
                FOREIGN KEY (calculation_id) REFERENCES calculations (id)
            )
        """)

        # Structures table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                formula TEXT,
                num_atoms INTEGER,
                cell_parameters TEXT,
                atomic_positions TEXT,
                created_date TEXT
            )
        """)

        # Adsorption energies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adsorption_energies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slab_id INTEGER,
                adsorbate TEXT,
                site_type TEXT,
                coverage REAL,
                adsorption_energy REAL,
                slab_energy REAL,
                adsorbate_energy REAL,
                total_energy REAL,
                calculation_date TEXT,
                FOREIGN KEY (slab_id) REFERENCES structures (id)
            )
        """)

        conn.commit()
        conn.close()

        return f"Database created successfully at {db_path}"

    except Exception as e:
        return f"Error creating database: {str(e)}"


@tool
def store_calculation(
    db_path: str,
    name: str,
    calculation_type: str,
    structure_file: str,
    input_parameters: Dict[str, Any],
    status: str = "pending",
    notes: Optional[str] = None,
) -> str:
    """Store calculation information in database.

    Args:
        db_path: Path to database file
        name: Calculation name/identifier
        calculation_type: Type of calculation (scf, relax, etc.)
        structure_file: Path to structure file
        input_parameters: Dictionary of input parameters
        status: Calculation status
        notes: Additional notes

    Returns:
        Storage status and calculation ID
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Insert calculation
        cursor.execute(
            """
            INSERT INTO calculations 
            (name, calculation_type, structure_file, input_parameters, 
             status, created_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                name,
                calculation_type,
                structure_file,
                json.dumps(input_parameters),
                status,
                datetime.now().isoformat(),
                notes,
            ),
        )

        calc_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return f"Calculation '{name}' stored with ID {calc_id}"

    except Exception as e:
        return f"Error storing calculation: {str(e)}"


@tool
def update_calculation_status(
    db_path: str,
    calculation_id: int,
    status: str,
    total_energy: Optional[float] = None,
    convergence_achieved: Optional[bool] = None,
) -> str:
    """Update calculation status and results.

    Args:
        db_path: Path to database file
        calculation_id: ID of calculation to update
        status: New status (running, completed, failed)
        total_energy: Total energy if available
        convergence_achieved: Whether calculation converged

    Returns:
        Update status
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Prepare update query
        update_fields = ["status = ?"]
        values = [status]

        if total_energy is not None:
            update_fields.append("total_energy = ?")
            values.append(total_energy)

        if convergence_achieved is not None:
            update_fields.append("convergence_achieved = ?")
            values.append(convergence_achieved)

        if status == "completed":
            update_fields.append("completed_date = ?")
            values.append(datetime.now().isoformat())

        values.append(calculation_id)

        # Execute update
        query = f"UPDATE calculations SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, values)

        conn.commit()
        conn.close()

        return f"Calculation {calculation_id} updated to status: {status}"

    except Exception as e:
        return f"Error updating calculation: {str(e)}"


@tool
def store_adsorption_energy(
    db_path: str,
    slab_name: str,
    adsorbate: str,
    site_type: str,
    coverage: float,
    adsorption_energy: float,
    slab_energy: float,
    adsorbate_energy: float,
    total_energy: float,
) -> str:
    """Store adsorption energy calculation results.

    Args:
        db_path: Path to database file
        slab_name: Name of slab structure
        adsorbate: Adsorbate species
        site_type: Adsorption site type
        coverage: Surface coverage
        adsorption_energy: Calculated adsorption energy
        slab_energy: Clean slab energy
        adsorbate_energy: Gas-phase adsorbate energy
        total_energy: Total system energy

    Returns:
        Storage status
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Find slab structure ID
        cursor.execute("SELECT id FROM structures WHERE name = ?", (slab_name,))
        result = cursor.fetchone()

        if not result:
            return f"Slab structure '{slab_name}' not found in database"

        slab_id = result[0]

        # Insert adsorption energy
        cursor.execute(
            """
            INSERT INTO adsorption_energies 
            (slab_id, adsorbate, site_type, coverage, adsorption_energy,
             slab_energy, adsorbate_energy, total_energy, calculation_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                slab_id,
                adsorbate,
                site_type,
                coverage,
                adsorption_energy,
                slab_energy,
                adsorbate_energy,
                total_energy,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

        return f"Adsorption energy stored for {adsorbate} on {slab_name}"

    except Exception as e:
        return f"Error storing adsorption energy: {str(e)}"


@tool
def query_calculations(
    db_path: str,
    status: Optional[str] = None,
    calculation_type: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Query calculation database.

    Args:
        db_path: Path to database file
        status: Filter by status
        calculation_type: Filter by calculation type
        limit: Maximum number of results

    Returns:
        Query results
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Build query
        query = "SELECT id, name, calculation_type, status, total_energy, created_date FROM calculations"
        conditions = []
        values = []

        if status:
            conditions.append("status = ?")
            values.append(status)

        if calculation_type:
            conditions.append("calculation_type = ?")
            values.append(calculation_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_date DESC LIMIT ?"
        values.append(limit)

        cursor.execute(query, values)
        results = cursor.fetchall()

        conn.close()

        if not results:
            return "No calculations found matching criteria"

        # Format results
        output = f"Found {len(results)} calculations:\n\n"
        for row in results:
            calc_id, name, calc_type, calc_status, energy, date = row
            energy_str = f"{energy:.6f} eV" if energy else "N/A"
            output += f"ID {calc_id}: {name} ({calc_type}) - {calc_status} - {energy_str} - {date}\n"

        return output

    except Exception as e:
        return f"Error querying calculations: {str(e)}"


@tool
def export_results(
    db_path: str, output_format: str = "json", calculation_ids: Optional[List[int]] = None
) -> str:
    """Export calculation results to file.

    Args:
        db_path: Path to database file
        output_format: Export format (json, csv)
        calculation_ids: Specific calculation IDs to export

    Returns:
        Export status and file path
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Build query
        if calculation_ids:
            placeholders = ",".join("?" * len(calculation_ids))
            query = f"SELECT * FROM calculations WHERE id IN ({placeholders})"
            cursor.execute(query, calculation_ids)
        else:
            cursor.execute("SELECT * FROM calculations")

        # Get column names
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()

        conn.close()

        # Prepare output
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format.lower() == "json":
            output_file = output_dir / f"calculations_export_{timestamp}.json"

            # Convert to list of dictionaries
            data = []
            for row in results:
                record = dict(zip(columns, row, strict=False))
                # Parse JSON fields
                if record.get("input_parameters"):
                    try:
                        record["input_parameters"] = json.loads(
                            record["input_parameters"]
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass
                data.append(record)

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

        elif output_format.lower() == "csv":
            output_file = output_dir / f"calculations_export_{timestamp}.csv"

            with open(output_file, "w") as f:
                # Write header
                f.write(",".join(columns) + "\n")

                # Write data
                for row in results:
                    # Convert to strings and escape commas
                    row_str = []
                    for item in row:
                        if item is None:
                            row_str.append("")
                        else:
                            str_item = str(item).replace(",", ";")
                            row_str.append(str_item)
                    f.write(",".join(row_str) + "\n")

        else:
            return f"Unsupported format: {output_format}"

        return f"Results exported to {output_file}"

    except Exception as e:
        return f"Error exporting results: {str(e)}"


@tool
def search_similar_calculations(
    db_path: str, reference_structure: str, calculation_type: str, tolerance: float = 0.1
) -> str:
    """Search for similar calculations in database.

    Args:
        db_path: Path to database file
        reference_structure: Reference structure file or formula
        calculation_type: Type of calculation to search for
        tolerance: Tolerance for similarity matching

    Returns:
        Similar calculations found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query calculations of the same type
        cursor.execute(
            """
            SELECT c.id, c.name, c.total_energy, s.formula, s.num_atoms
            FROM calculations c
            LEFT JOIN structures s ON c.structure_file LIKE '%' || s.name || '%'
            WHERE c.calculation_type = ? AND c.status = 'completed'
        """,
            (calculation_type,),
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return f"No completed {calculation_type} calculations found"

        # Simple similarity based on formula/structure
        similar = []
        for calc_id, name, energy, formula, num_atoms in results:
            if reference_structure in str(name) or reference_structure in str(formula):
                similar.append((calc_id, name, energy, formula, num_atoms))

        if not similar:
            return f"No similar calculations found for {reference_structure}"

        # Format results
        output = f"Found {len(similar)} similar calculations:\n\n"
        for calc_id, name, energy, formula, num_atoms in similar:
            energy_str = f"{energy:.6f} eV" if energy else "N/A"
            output += (
                f"ID {calc_id}: {name} - {formula} ({num_atoms} atoms) - {energy_str}\n"
            )

        return output

    except Exception as e:
        return f"Error searching calculations: {str(e)}"
