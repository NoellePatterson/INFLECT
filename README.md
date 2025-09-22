# INFLection-based Elevations from Channel Topography (INFLECT)

INFLECT is a software tool used to identify persistent topographic features of river corridors, such as bankfull and flood terraces. INFLECT uses cross-sections derived from digital elevation models to identify abrupt changes in cross-section width, which are indicative of topographic features. Inputs to INFLECT are a DEM, a river thalweg/centerline, and a set of transects spanning the study reach. Outputs include a series of diagnostic figures described below, a set of elevations representing major topographic feature locations on the study reach, and a shapefile of topographic feature locations along the input transects. 

## Software Configuration

The following steps outline how to run the INFLECT tool and all dependencies required, for both Windows and Mac operating systems. 

### 1. Install Python 3

#### Windows

1. Download the latest version of Python 3 from <python.org/downloads>
2. Run the installer:
- On the first screen, check “Add Python 3.x to PATH”.
- Click Install Now.
3. Verify the installation by opening Command Prompt (`Win + R`, type `cmd`) and running:
  ```
  python --version
  ```
You should see a version starting with 3. (e.g., Python 3.12.1).

#### Mac

1. Open **Terminal**.
2. Check if Python 3 is already installed:
```
python --version
```
- If you see `Python 3.x.x`, you’re good to go.
- If not, install it using [Homebrew](https://brew.sh/) (recommended):
    ```
    brew install python
    ```
3. Verify the installation:
   ```
   python3 --version
   ```

### 2. Clone the Repository

In your command line (Command Prompt on Windows, Terminal on Mac):
```
git clone https://github.com/noelle_patterson/INFLECT.git
cd INFLECT
```

### 3. Create a Virtual Environment

This step keeps dependencies organized.

#### Windows
```
python -m venv venv
venv\Scripts\activate
```

#### Mac
```
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Once the virtual environment is activated, install required packages:
```
pip install -r requirements.txt
```

## Running the Code
