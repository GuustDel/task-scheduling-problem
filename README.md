# Production Line Worker Allocation CSP

This project is a constraint programming application that uses [CPMpy](https://github.com/CPMpy/cpmpy) and Tkinter to allocate workers along a production line for integrated solar panels. The application minimizes the total number of manual workers required while ensuring that each manual task runs at a pace set by the bottleneck (slowest) automated task. It also includes a soft constraint to favor preferred workers—helpful when managing multiple projects on the same day.

## Features

- **Constraint Optimization:** Balances manual tasks’ effective times with the automated bottleneck using CPMpy.
- **Integer-Only Arithmetic:** Converts cycle times into integer fractions (e.g., 4.5 is stored as 9/2) to comply with CPMpy’s requirements.
- **Worker Skill Matrix:** Specify which workers can perform which manual tasks.
- **Worker Preferences:** Mark workers as preferred so that, when solving multiple projects, the system favors using the same workers.
- **Graphical User Interface:** A Tkinter-based UI for entering parameters and viewing results side-by-side.

## Prerequisites

- **Python 3.7+**
- **Tkinter:** Typically included with standard Python installations. (On some Linux distributions you might need to install it separately, e.g., via `sudo apt-get install python3-tk`.)
- **CPMpy**
- **NumPy**

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/production-line-worker-allocation.git
   cd production-line-worker-allocation
   ```

2. **(Optional) Create a virtual environment:**

   - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**

   ```bash
   python your_script_name.py
   ```

Replace your_script_name.py with the name of the Python file containing the application code.

2. **Enter Parameters:**

- **Automated Task Cycle Times:** Input the cycle times for the automated tasks (these determine the bottleneck production rate).
- **Manual Task Base Times:** Enter the base times for each manual task (as floats, e.g., 4.5 for 9/2).
- **Worker Skill Matrix:** Use the checkboxes to indicate which workers are qualified for each manual task.
- **Worker Preferences:** Check the boxes next to preferred workers to favor their use in the solution.

3. **Solve the Allocation:**

Click the **Solve Allocation** button. The application will run the constraint model and display:

- The bottleneck cycle time.
- The total number of workers used.
- A detailed assignment for each manual task (including the effective time and the list of assigned workers).
- An overall summary of worker usage and preference status.

## Customization

- **Task and Worker Settings:** You can modify the number of tasks, default times, or the number of workers directly in the source code.
- **Objective Weights:** To adjust how strongly the model favors preferred workers, change the `LARGE_WEIGHT` constant and the objective formulation in the solve_model function.

## License

This project is open source under the MIT License. See the `LICENSE` file for details.

## Contributions

Feel free to open an issue or submit a pull request if you have any improvements or suggestions!

## Contact

For any questions or suggestions, please open an issue or contact the repository owner.