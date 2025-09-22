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

Once you have configured the repository on your local machine, you are ready to start running the code and generating results. 

### 1. Upload user data

Add data for your study river reach to the `data_inputs` folder within the repository. There are three total data needs:

1. Add a **DEM** to the `data_inputs/dem` folder, in .TIF file format.
   
INFLECT has been developed and tested using a 1-m resolution DEM. It is highly recommended to use this resolution or similar with INFLECT. The DEM must cover the full extent of the thalweg and cross-section data inputs. 

2. Add a **thalweg line** to `data_inputs/thalweg` folder, as a line type shapefile in .shp file format.

The accuracy of the thalweg line is not critical to INFLECT performance, and a river centerline may be used instead. The length of river reach used in INFLECT is an ongoing research question and the user is encouraged to inspect results carefully when selecting a river segment and length. INFLECT has been developed with 1-5 km river reaches and this length is encouraged as a starting point. It is also crucial to avoid abrupt drops in the longitudinal profile of the reach, such as those caused by knickpoints or waterfalls, as these are not well-suited to the linear detrending performed in INFLECT.

3. Add **cross-sections** to the `data_inputs/cross_sections` folder, as a line type shapefile in .shp file format.
  
Cross-sections can be automatically generated from the input thalweg. Cross-sections must be ordered from upstream to downstream. Cross-section width may be varied to suit the study site, but should cover the river corridor width of interest plus further extent up the valley walls if possible. It is okay if cross-sections overlap. Cross-section spacing can be determined by the user and should be spaced closely enough to capture the occurrence of both riffles and pools. Spacing of 10-50 meters have been tested successfully in INFLECT. Cross-sections do not need to be evenly spaced. 
