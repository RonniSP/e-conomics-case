%% e-conomics case Part 1 - Per company prediction

%  Description
%  ------------
%  This file contains code that implements a model for predicting the
%  accounts to be used based on training done on the data for the
%  individual companies.

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading the data =============
%  We start by first loading the dataset and creating a table with the
%  data we need. This includes "CompanyId" in order to identify the
%  individual company; "AccountNumber" in order to identify the account
%  used; "BankEntryText" which is the data describing the transaction.

% Load Data into a single table
fprintf('Loading Data ...\n')

formatSpec = '%d%s%{yyyy-mm-dd}D%s%s%s%d%s';
data_table = readtable('bank_expenses_obfuscated.csv', 'Format', formatSpec);

% Create column vectors for "CompanyId", "AccountNumber" and
% "BankEntryText"
companyId = data_table(:, 2);
companyId = table2array(companyId);
accountNumber = data_table(:, 7);
accountNumber = table2array(accountNumber);
bankEntryText = data_table(:, 4);
bankEntryText = table2array(bankEntryText);

fprintf('Data ready ...\n')

%% =========== Part 2: Preperation of usefull data =============
% Data required for making the prediction model is prepared before creating
% the models
fprintf('Preparing data ...\n')
% Make a list of all company id's
all_companies = unique(companyId);

% Locate the first and last row for each company in the dataset and ad this 
% information to a list
first_occurrence = zeros(size(all_companies, 1), 1);
last_occurrence = zeros(size(all_companies, 1), 1);
tracker = 0;

for i = 1:size(all_companies)
    company = all_companies(i);
    temp_index = find(cellfun('length', regexp(companyId, company)) == 1);
    first_occurrence(i) = min(temp_index);
    last_occurrence(i) = max(temp_index);
    tracker = tracker + 1;
    
    fprintf('%d out of %d prepared ...\n', i, size(all_companies, 1))
end

%% ======= Part 3: Making the One-VS-All models and testing them =========
% Using a for-loop we go through all companies and make a model for each of
% them
all_models = struct;%(size(all_companies,1));
test_results = zeros(size(all_companies, 1), 1);
% Start the for-loop that goes through all companies
for i = 74:size(all_companies)
    company = all_companies(i);

% Fetch all bank entries for the company and the respective account used.
% Also find all unique accounts and the total number of unique accounts.
    temp_accountNumber = accountNumber(first_occurrence(i):last_occurrence(i));
    temp_bankEntryText = bankEntryText(first_occurrence(i):last_occurrence(i));
    accounts = unique(temp_accountNumber);
    account_count = size(accounts, 1);
    
% Make a list of every string and integer combination that occurs in 
% temp_bankEntryText and find the largest number of entries occuring in one
% entry.
    unique_entries = [];
    Highest_number_of_entries = 0;
    for idx = 1:size(temp_bankEntryText)
        text = cell2mat(temp_bankEntryText(idx));
        text = strsplit(text);
        unique_entries = [unique_entries, text];
        unique_entries = unique(unique_entries);
        if size(text, 2) > Highest_number_of_entries
            Highest_number_of_entries = size(text, 2);
        end
        fprintf('List entry nr. %d out of %d prepared ...\n', idx, size(temp_bankEntryText, 1));
    end    
    unique_entries = unique_entries';
    
    
% Make a matrix where each row is the string and integer combination
% that occurs in the same row number in temp_bankEntryText defined by 
% their unique number. The number of columns should be equal to the largest 
% number of entries in any of the rows.
    temp_entry_ID = zeros(size(temp_bankEntryText, 1), Highest_number_of_entries);
    for idx = 1:size(temp_bankEntryText)
        text = cell2mat(temp_bankEntryText(idx));
        text = strsplit(text);
        entry_fingerprint = [];
        for entry = 1:size(text, 2)
            temp_index = find(cellfun('length', regexp(unique_entries, text(entry))) == 1);
            entry_fingerprint = [entry_fingerprint, temp_index];
        end
        temp_entry_ID(idx, 1: size(text, 2)) = entry_fingerprint;
    end
    
% Run One-Vs-All to find the optimal models based on this prediction
    lambda = 0.5;
    [all_theta] = oneVsAll(temp_entry_ID, temp_accountNumber, account_count,accounts , lambda);
    all_models(i).model = all_theta;


% Divide the dataset into a training set (2/3) and a test set (1/3)

% Test the prediction of the One-Vs-All classifier for this company    
    pred = predictOneVsAll(all_theta, temp_entry_ID);
    pred_account = zeros(size(pred, 1), 1);
    for idx = 1:size(pred, 1)
        pred_account(idx) = accounts(pred(idx));
    end
    test_results(i) = mean(double(pred_account == temp_accountNumber)) * 100;
    fprintf('Testing of company nr. %d out of %d prepared ...\n', i, size(all_companies, 1));
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_account == temp_accountNumber)) * 100);

    

end











