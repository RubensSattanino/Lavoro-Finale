% Loading data
data = load('SP.txt');
data = data .* 100;
Ndata = normalize(data);
realData = arrayDatastore(Ndata);
% Specify training options
options = struct('miniBatchSize', 128, ...
                 'maxEpochs', 100, ...
                 'learningRateSchedule', 'decreasing', ...
                 'learningRate', 0.002, ...
                 'LearnRateDropFactor', 0.1, ...
                 'limitLearnRate', 0.0001, ...
                 'LearnRateDropPeriod', 50, ...
                 'gradientDecayFactor', 0.05, ...
                 'squaredGradientDecayFactor', 0.999);

miniBatchSize = options.miniBatchSize;
numSamples = size(data, 2);

% Definition of the Generator Neural Network
Generator = [
    inputLayer([miniBatchSize numSamples], 'SC')
    fullyConnectedLayer(128)
    leakyReluLayer(0.2)
    fullyConnectedLayer(256) 
    leakyReluLayer(0.2)
    fullyConnectedLayer(300) 
    leakyReluLayer(0.2)
    dropoutLayer(0.1)
    fullyConnectedLayer(miniBatchSize)
    tanhLayer
]; 

netG = dlnetwork(Generator);

% Definition of the Discriminator Neural Network
Discriminator = [
    inputLayer([miniBatchSize numSamples], 'SC')
    fullyConnectedLayer(128)
    leakyReluLayer(0.2)
    fullyConnectedLayer(256)
    leakyReluLayer(0.2)
    fullyConnectedLayer(300) 
    leakyReluLayer(0.2)
    fullyConnectedLayer(miniBatchSize) 
    sigmoidLayer
];
netD = dlnetwork(Discriminator); 

trainGan(realData, netG, netD, options)

function trainGan(data, netG, netD, options)
    % Extracting options
    miniBatchSize = options.miniBatchSize;
    maxEpochs = options.maxEpochs;
    learningRateSchedule = options.learningRateSchedule;
    learningRate = options.learningRate;
    LearnRateDropFactor = options.LearnRateDropFactor;
    limitLearnRate = options.limitLearnRate;
    LearnRateDropPeriod = options.LearnRateDropPeriod;
    gradientDecayFactor = options.gradientDecayFactor;
    squaredGradientDecayFactor = options.squaredGradientDecayFactor;

    numSamples = size(data, 2);
    
    % Monitoring training progress
    monitor = trainingProgressMonitor( ...
        Metrics=["GeneratorLoss","DiscriminatorLoss"], ...
        Info=["Epoch", "Iteration", 'LearningRateSchedule', 'LearningRate'], ...
        XLabel="Iteration", ...
        Status = 'Running');

    % Create minibatch queues for training, validation, and test datasets
    mbq = minibatchqueue(data, ...
        MiniBatchSize=miniBatchSize, ...
        PartialMiniBatch="discard", ...
        MiniBatchFormat="SC");
   
    % Initialize parameters for Adam optimization algorithm
    [trailingAvgG, trailingAvgSqG, trailingAvg, trailingAvgSqD] = deal([]);
    
    % Training loop
    epoch = 0;
    iteration = 0;  
    
    while epoch < maxEpochs && ~monitor.Stop
        epoch = epoch + 1;
       
        % Reset and shuffle datastore
        shuffle(mbq);
    
        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;
            
            learningRate = adjustLearningRate(learningRate, ...
                learningRateSchedule, ...
                LearnRateDropFactor, limitLearnRate, LearnRateDropPeriod, iteration);
  
            % Read mini-batch of data
            X = next(mbq);

            % Generate latent inputs for the generator network
            Z = randn([miniBatchSize numSamples], "single");
            Z = dlarray(Z, "SC");
    
            if canUseGPU
               Z = gpuArray(Z);
            end
    
            % Compute losses and gradients using the modelLoss function
            [lossG, lossD, gradientsG, gradientsD, stateG] = ...
                dlfeval(@modelLoss, netG, netD, X, Z);
            netG.State = stateG;
    
            % Update the discriminator network parameters
            [netD, trailingAvg, trailingAvgSqD] = adamupdate(netD, gradientsD, ...
                trailingAvg, trailingAvgSqD, iteration, ...
                learningRate, gradientDecayFactor, squaredGradientDecayFactor);
    
            % Update the generator network parameters
            [netG, trailingAvgG, trailingAvgSqG] = adamupdate(netG, gradientsG, ...
                trailingAvgG, trailingAvgSqG, iteration, ...
                learningRate, gradientDecayFactor, squaredGradientDecayFactor);

            % Update training progress
            updateTrainingProgress(monitor, iteration, ...
                epoch, maxEpochs, ...
                learningRateSchedule, ...
                learningRate, lossG, lossD)
        end
    end
    
    
    
 
    
    % Function for adjusting learning rate
   function learning_rate = adjustLearningRate(initial_rate, LearnRateSchedule, LearnRateDropFactor, limitLearnRate, LearnRateDropPeriod, iteration)
        % Set default values if not provided
        if nargin < 2, LearnRateSchedule = 'constant'; end
        if isempty(LearnRateDropPeriod), LearnRateDropPeriod = 1; end
        if isempty(LearnRateDropFactor), LearnRateDropFactor = 0; end
        if isempty(limitLearnRate), limitLearnRate = []; end
    
        % Default learning rate
        learning_rate = initial_rate;
    
        % Adjust learning rate based on schedule
        if mod(iteration, LearnRateDropPeriod) == 0 && iteration > 0
            switch lower(LearnRateSchedule)
                case 'decreasing'
                    learning_rate = initial_rate * (1 - LearnRateDropFactor);
                    if ~isempty(limitLearnRate)
                        learning_rate = max(learning_rate, limitLearnRate);
                    end
                case 'increasing'
                    learning_rate = initial_rate * (1 + LearnRateDropFactor);
                    if ~isempty(limitLearnRate)
                        learning_rate = min(learning_rate, limitLearnRate);
                    end
                % 'constant' and other cases fall back to initial_rate
            end
        end
    end

    % Function for updating training progress
    function updateTrainingProgress(monitor, iteration, epoch, maxEpochs, learningRateSchedule, learningRate, lossG, lossD)
        % Computing the number of iterations
        numObservationsTrain = numel(readall(data));
        numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
        numIterations = maxEpochs * numIterationsPerEpoch;
    
        % Grouping subplots for loss metrics
        groupSubPlot(monitor, Loss=["GeneratorLoss", "DiscriminatorLoss"]);
        
        % Update the training progress monitor
        recordMetrics(monitor, ...
            iteration, ...
            GeneratorLoss=lossG, ...
            DiscriminatorLoss=lossD);
        
        % Update information displayed on the monitor
        updateInfo(monitor, ...
            Epoch = string(epoch) + ' of ' + string(maxEpochs), ...
            Iteration = string(iteration) + ' of ' + string(numIterations), ...
            LearningRateSchedule = learningRateSchedule, ...
            LearningRate = learningRate);
    
        % Update progress percentage
        monitor.Progress = iteration / numIterations * 100;
        if iteration == numIterations
            monitor.Status = 'Max epochs completed';
        elseif monitor.Stop
            monitor.Status = 'Training stopped';
        end
    end

    % Model Loss Function
    function [lossG, lossD, gradientsG, gradientsD, stateG, XGenerated] = modelLoss(netG, netD, X, Z)
        % Calculate the predictions for real data with the discriminator network
        YReal = forward(netD, X);
        
        % Calculate the predictions for generated data with the discriminator network
        [XGenerated, stateG] = forward(netG, Z);
        XGenerated = dlarray(XGenerated, 'SC');
    
        YGenerated = forward(netD, XGenerated);
       
        % Calculate the GAN loss
        [lossG, lossD] = ganLoss(YReal, YGenerated);
         
        % For each network, calculate the gradients with respect to the loss
        gradientsG = dlgradient(lossG, netG.Learnables, RetainData=true);
        gradientsD = dlgradient(lossD, netD.Learnables, RetainData=true);
    end
    
    % GAN loss function
    function [lossG, lossD] = ganLoss(YReal, YGenerated)
        % Binary Cross-Entropy Loss for Discriminator
        lossD = - (mean(log(YReal)) + mean(log(1 - YGenerated)));
        lossG = - mean(log(YGenerated));
    end
end
