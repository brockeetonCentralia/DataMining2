using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;

namespace DataMining2
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private string testDataPath;
        private IDataView testData;
        private IDataView trainingData;
        private string modelPath;
        private ITransformer trainedModel;
        private EstimatorChain<KeyToValueMappingTransformer> pipeline;
        private bool IsRunning = true;

        //prediction engine (input and output types)
        public Program()
        {
            //gather any variables and set them
            dataPath = Path.Combine(Environment.CurrentDirectory, "data\\DisneylandReviews.csv");
            testDataPath = Path.Combine(Environment.CurrentDirectory, "data\\TestData.csv");
            modelPath = Path.Combine(Environment.CurrentDirectory, "models", "model");

            //create a context (connection to the database)
            ctx = new MLContext();

            //create a query to get the data
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true, separatorChar: ',');

            //build data pipeline (transforming your data into something that works)
            CreatePipeline();

            TrainAndSaveModel();

            while (IsRunning)
            {
                Console.Clear();

                Console.WriteLine("---------------------");
                Console.WriteLine("Main Menu");
                Console.WriteLine("---------------------");
                Console.WriteLine("1. Retrain Model");
                Console.WriteLine("2. Evaluate Model");
                Console.WriteLine("3. Predict");
                Console.WriteLine("4. Exit");

                int choice = -1;
                Console.WriteLine("Enter your choice: ");
                choice = int.Parse(Console.ReadLine());

                switch (choice)
                {
                    case 1:
                        TrainAndSaveModel();
                        Console.WriteLine("Model has been retrained and saved. Press enter to continue.");
                        Console.ReadLine();
                        break;
                    case 2:
                        EvaluateModel();
                        Console.WriteLine("Press enter to continue...");
                        Console.ReadLine();
                        break;
                    case 3:
                        MakePrediction();
                        Console.WriteLine("Press enter to continue...");
                        Console.ReadLine();
                        break;
                    case 4:
                        IsRunning = false;
                        break;
                    default:
                        Console.WriteLine("Invalid choice: press enter to try again");
                        Console.ReadLine();
                        break;
                }
            }
        }

        private void MakePrediction()
        {
            Console.WriteLine("Enter a review: ");
            string reviewText = Console.ReadLine();

            DisneylandReview review = new DisneylandReview()
            {
                ReviewText = reviewText
            };

            var prediction = Predict(review);

            PrintPrediction(prediction, review);
        }

        private void EvaluateModel()
        {
            var schema = trainingData.Schema;

            testData = ctx.Data.LoadFromTextFile<DisneylandReview>(testDataPath, hasHeader: true, separatorChar: ',');

            var testMetrics = ctx.MulticlassClassification.Evaluate(trainedModel.Transform(testData));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

        }

        private void TrainAndSaveModel()
        {
            trainedModel = pipeline.Fit(trainingData);
            SaveModel(ctx, trainingData.Schema, trainedModel);
        }

        private void CreatePipeline()
        {
             var pipeline = ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Rating", outputColumnName: "Label")
            .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "ReviewText", outputColumnName: "FeaturizedReviewText"))
            .Append(ctx.Transforms.Concatenate("Features", "FeaturizedReviewText"))
            .AppendCacheCheckpoint(ctx)
            .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }
        public void SaveModel(MLContext mLContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mLContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public void PrintPrediction(DisneylandPrediction prediction, DisneylandReview review)
        {
            Console.WriteLine($"Review Text: {review.ReviewText}" +
                 $"\nPredicted rating: {prediction.Label}" +
                 $"\nActual Rating: {review.Rating}");

            for (int i = 0; i < prediction.Score.Length; i++)
            {
                Console.WriteLine($"Score {i + 1}: {prediction.Score[i]}");
            }
        }

        public DisneylandPrediction Predict(DisneylandReview review)
        {
            var predictionEngine = ctx.Model.CreatePredictionEngine<DisneylandReview, DisneylandPrediction>(trainedModel);
            var resultPrediction = predictionEngine.Predict(review);
            return resultPrediction;
        }

        static void Main(string[] args)
        {
            new Program();
        }
    }
}
