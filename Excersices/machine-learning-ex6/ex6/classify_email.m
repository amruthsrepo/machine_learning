% Classify email

load('emailTrainedModel.mat');
filename = 'testEmail.txt';

file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %s (%d)\n', filename, ifelse(p,'Spam','Not Spam'), p);
