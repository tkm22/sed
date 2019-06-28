CREATE TABLE `Utility Rates` (
  `Year` int,
  `ZIPCode` int,
  `State` varchar(200),
  `UtilityCompany` varchar(200),
  `Ownership` varchar(200),
  `ServiceType` varchar(200),
  `ResidentialRate` float,
  `CommercialRate` float,
  `IndustrialRate` float,
  PRIMARY KEY (`Year`, `ZIPCode`)
);

CREATE TABLE `Household Income by Census Block` (
  `Year` int,
  `CensusBlock` int,
  `City` varchar(200),
  `State` varchar(200),
  `HouseholdIncome` int,
  `HouseholdIncomeMedian` int,
  PRIMARY KEY (`Year`, `CensusBlock`)
);

CREATE TABLE `Energy Consumption` (
  `UID` int,
  `BuildingID` int,
  `Year` int,
  `ZIPCode` int,
  `City` varchar(200),
  `State` varchar(200),
  `BuildingType` varchar(200),
  `PrimaryPropertyUse` varchar(200),
  `YearBuilt` int,
  `PropertyGFA(sf)` int,
  `SiteEUI(kBtu/sf)` float,
  `SiteEUIWN(kBtu/sf)` float,
  `SourceEUI(kBtu/sf)` float,
  `SourceEUIWN(kBtu/sf)` float,
  PRIMARY KEY (`UID`)
);

CREATE TABLE `ZIP Code to Census Block` (
  `ZIPCode` int,
  `CensusBlock` int,
  PRIMARY KEY (`ZIPCode`)
);
