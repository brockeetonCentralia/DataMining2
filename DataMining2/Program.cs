using Microsoft.ML;
using System;

namespace DataMining2
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private string testDataPath;
        private IDataView trainingData;
        private string modelPath;
        private ITransformer trainedModel;

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

            var pipeline = ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Rating", outputColumnName: "Label")
                .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "ReviewText", outputColumnName: "FeaturizedReviewText"))
                .Append(ctx.Transforms.Concatenate("Features", "FeaturizedReviewText"))
                .AppendCacheCheckpoint(ctx)
                .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            //train your model (make it run the data)
            trainedModel = pipeline.Fit(trainingData);

            //save the model
            SaveModel();

            //capture some text
            //var sampleStatement = new DisneylandReview
            //{
            //    ReviewId = 2842749,
            //    Rating = "1",
            //    YearMonth = "missing",
            //    ReviewerLocation = "United Kingdom",
            //    ReviewText = "I will never go to disneyland again. The staff were rude, the park was dirty, and the rides were broken. It was a waste of money.",
            //    Branch = "Disneyland_Paris"
            //};

            //var sampleStatement2 = new DisneylandReview
            //{
            //    ReviewId = 2801260,
            //    Rating = "2",
            //    YearMonth = "missing",
            //    ReviewerLocation = "United Kingdom",
            //    ReviewText = "We've just come back from a couple of days at Eurodisney. I agree with a previous review writer that to the staff at Eurodisney its just a job, whereas at Orlando they do the job because they actually like to do it. That's not belittling all the staff at Paris as some where very helpful, albeit they seemed in the minority.We weren't warned that some of the rides were closed and the fast path tickets ran out early afternoon. Additionally we couldn't get a Park Guide in English as they had all ran out! We booked our tickets over the internet direct with Disney which failed to arrive and we waited 40 minutes to be given replacements on the day. Having been to Disney at Florida early this year perhaps gave us an unfair comparison.Whilst writing   does anyone know what the Halloween song was that was played?We didn't feel that the Park had the same Disney feel as Florida, true it was Halloween there but you could have been anywhere. There were very few characters going around.Having said all that   of course we will go again!!!",
            //    Branch = "Disneyland_Paris"
            //};

            //var prediction = Predict(sampleStatement);

            //PrintPrediction(prediction, sampleStatement);

            Console.ReadLine();
        }

        public void SaveModel()
        {
            ctx.Model.Save(trainedModel, trainingData.Schema, modelPath);
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
