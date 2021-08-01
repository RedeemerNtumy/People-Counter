# People Counter

This is a Computer Vision program that counts the number of people entering and exiting a building.



## HOW TO RUN (WINDOWS)

1. Download and install [Visual C++ build tools](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)

2. Download and install [CMake](https://cmake.org/download/) and add it to the environment variables.

3. Navigate to the project's base directory.

4. Install the required python 3 modules using the command:

   `python -m pip install -r requirements.txt`

   *NB:  The system has been tested using Python 3.7.4 to 3.8.5*

5. To start the web server, navigate to the `/website/` directory and run the command: 

   `flask run`

6. To start the desktop application, navigate the `/desktop-app/` directory and run the command: 

   `python alpha_main.py`

   


  ###### FEATURES

  - Counts various individuals moving in and out of the building.
  - Dashboard that shows the number of people entering and leaving in real time
  - Video integrated into dashboard for live preview of people entering and leaving
  - Database which saves the number of people entering and leaving and the timestamps
  - A data table which shows the the daily number of people who entered the building
 
  
