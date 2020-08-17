[![Build Status](https://travis-ci.com/kmpg-capstone-2019/sed.svg?branch=master)](https://travis-ci.com/kmpg-capstone-2019/sed)
[![Coverage Status](https://coveralls.io/repos/github/kmpg-capstone-2019/sed/badge.svg?branch=master)](https://coveralls.io/github/kmpg-capstone-2019/sed?branch=master)

## Signals for Energy Demand

Signals for Energy Demand (sed) is a capstone project by DIRECT trainees in collaboration with KPMG. 
The project is meant to allow the DIRECT trainees to cement the acquisition of data science skills and develop proficiency in the
conduct of team-based interdisciplinary research, while supporting the needs of their collaborators.

### Project Description

Understanding and forecasting domestic and commercial energy demand is of great concern to utility
companies, facilities managers, and building commissioning projects for energy-saving initiatives. Utility
companies use demand estimates to reduce operating costs by ensuring the right amount of energy is
produced, avoiding wasteful extra energy production or costly outages. Locally, facilities and operations
managers make plans to optimize the operations of chillers, boilers, and energy storage systems. While
energy demand is largely cyclical, is it also highly dependent on a number of climatic, economic, geospatial,
and demographic variables that are publicly available. Understanding the role of these local variables in driving
energy demand can guide future infrastructure investment and also feed-back into predictive mores for
improved forecasting accuracy and can serve as a method to predict local energy demand. 

This project utilizes existing publicly-available and KPMG Signals Repository datasets of energy
consumption and local economic, geospatial, and demographic variables to develop statistical insights and
Machine Learning models around local consumer or business energy consumption. Insights from this model
will be used to understand and predict local energy demand, and may potentially be used to as a
general method to forecast local energy consumption for future KPMG Advisory Engagements.

### Project Poster
![Poster]

[Poster]: https://github.com/tkm22/sed/blob/master/doc/KPMG_Poster.jpg

   


### Organization of the project

The project has the following structure:

    sed/
      |- README.md
      |- sed/
         |- __init__.py
         |- sed.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
      |- setup.py
      |- .travis.yml
      |- .mailmap
      |- appveyor.yml
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...

