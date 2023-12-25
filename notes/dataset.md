# Info about the Dataset
### General
- two datasets, identity and transaction
    - identity:
        - features: 41
        - records: 144233
    - transaction:
        - features: 394
        - records: 590540
- transaction ids are all unique in transactions but some of the transactions are linked to additional information in identity
  
### Identity
- Transaction ID -> matching part of the ids from the id dataset, only unique values
    - each Transaction ID in identity is also in transaction
- id_30 operating system of device
- id_31 browser of device
- id_33 display resolution of device
- id_35-id_38 boolean values unclear what they contain
- DeviceType (mobile or desktop)
- Device Info (has different contents a lot some are the pure OS (windows, macOS, etc..) some contain detailed information about the device build itself)

### Transaction
- Transaction ID -> not every transaction ID is mapped to one identity information from the other dataset
- isFraud -> binary target, 0 or 1
- TransactionDT -> some time value unclear what exactly but mostly increasing with increasing Transaction ID
    - also comparing the train to the test data it seems that the test data was recorded after the train data
- productCD -> has only values [W, H, C, S, R]
- card1-card6 -> information about the card uses
    - card4 provider of card (visa, master, etc.)
    - card6 card type used (credit, debit, etc.)
- P_emaildomain -> email provider of user (POSSIBLY?) (gmail.com, yahoo.com, etc.)
- R_emaldomnain -> email provider of ? also contains (gmail.com, yahoo.com, etc. (also foreign adresses))
- C1 - C14 -> all numerical values
- D1 - D15 -> all numerical values
- M1 - M9 -> boolean values (T,F)
- V1 -V339 -> all numerical values

### Correlation
- maximum correlation between isFraud and a base column is between isFraud and V257 (0.383)
- correlation high mainly between isFraud and the V columns
- C Features are almost all highly correlated to eachother
    - except C3 which does not seen to correlate with any other C feature
- D Features are also highly correlated to eachother but mainly negatively correlated
    - Reason?
    - D11 also seems less correlated to the other feachers compared to any other D feature
- V features are mostly uncorrelated but have some correlations in subgroups