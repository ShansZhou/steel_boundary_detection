using System;
using System.Diagnostics;
using Emgu.CV.Structure;
using Emgu.CV;
using System.Collections;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV.UI;

namespace testDLL
{
    class Program
    {
        public static String[] GetFilesFrom(String searchFolder, String[] filters, bool isRecursive)
        {
            List<String> filesFound = new List<String>();
            var searchOption = isRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var filter in filters)
            {
                filesFound.AddRange(Directory.GetFiles(searchFolder, String.Format("*.{0}", filter), searchOption));
            }
            return filesFound.ToArray();
        }
        /* conversion */
        //convert image to bytearray
        public static byte[] imgToByteArray(Image img)
        {
            using (MemoryStream mStream = new MemoryStream())
            {
                img.Save(mStream, img.RawFormat);
                return mStream.ToArray();
            }
        }
        //convert bytearray to image
        public static Image byteArrayToImage(byte[] byteArrayIn)
        {
            using (MemoryStream mStream = new MemoryStream(byteArrayIn))
            {
                return Image.FromStream(mStream);
            }
        }


        /* MAIN */
        static void Main(string[] args)
        {

            //String searchFolder = @"D:\工业视觉项目\BS_翘扣头\_QKT\testEmgu1\testEmgu1\TestIMG/R12/";
            String searchFolder = @"D:\工业视觉项目\BS_翘扣头\_QKT\testEmgu1\testEmgu1\TestIMG/outlier/";
            //String searchFolder = @"C:\Users\Zhigonghui\source\repos\testDLL\testDLL\imgs";
            var filters = new String[] { "jpg","png" };
            var files = GetFilesFrom(searchFolder, filters, false);

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            TimeSpan ts = stopWatch.Elapsed;

            // USING EMGU.CV DEMO
            for (int num = 0; num < files.Length; num++)
            {
                Console.WriteLine(num.ToString());
                Image<Gray, byte> img_input = new Image<Gray, byte>(files[num].ToString()); // read emgu image from path
                
                Bitmap masterImage = new Bitmap(files[num].ToString()); // read drawing image from path

                //img_input = img_input.Flip(Emgu.CV.CvEnum.FlipType.Horizontal);
                //masterImage.RotateFlip(RotateFlipType.Rotate180FlipY);


                Image i = masterImage;
                ImageConverter converter = new ImageConverter();

                // convert drawing image to byte[]
                byte[] byte3 = imgToByteArray(i);
                byteArrayToImage(byte3);
                detectQKTDistrubuted mydeteect = new detectQKTDistrubuted();
                ArrayList results_detection = mydeteect.DoDetection(byte3); // resulting image, bw image, points location

                /* RESULT IMAGE FROM BYTE */
                Image img_measured = byteArrayToImage((byte[])results_detection[0]);
                Image img_measured1 = byteArrayToImage((byte[])results_detection[1]);

                //Image<Gray, byte> img_measured = (Image<Gray, byte>)results_detection[0]; // cast Typle : Image<Gray, byte>
                //Image<Gray, byte> img_measured1 = (Image<Gray, byte>)results_detection[1];
            
                ArrayList points = (ArrayList)results_detection[2]; // cast Typle : ArrayList

                ArrayList stats = (ArrayList)results_detection[3];
                /* TEST RESULTS */
                /*
                Image<Gray, Byte> normalizedMasterImage = new Image<Gray, Byte>((Bitmap)img_measured);
                for (int t = 0; t < points.Count; t++)
                {
                    MCvScalar color = new MCvScalar(255,0,0);
                    Point thispoint = (Point)points[t];
                    CvInvoke.Circle(normalizedMasterImage, thispoint, 10,color);
                }
                */

                /* SAVE EMGU IMAGE */
                Bitmap bmpTif = new Bitmap(img_measured);
                Image<Gray, Byte> normalizedMasterImage = new Image<Gray, Byte>((Bitmap)img_measured);

                Bitmap bmp = bmpTif.Clone(new Rectangle(0, 0, normalizedMasterImage.Cols, normalizedMasterImage.Rows),
                    System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                Graphics g = Graphics.FromImage(bmp);
                g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

                //g.DrawString(stats[0].ToString(), new Font("Tahoma", 50, FontStyle.Bold), Brushes.White, new Point(50, 100));

                //bmp.Save(@"D:\工业视觉项目\BS_翘扣头\_QKT\testEmgu1\testEmgu1\TestIMG/R12/out/" + num.ToString() + "_stats.png");
                bmp.Save(@"D:\工业视觉项目\BS_翘扣头\_QKT\testEmgu1\testEmgu1\TestIMG/outlier/out/" + num.ToString() + "_stats.png");

                //img_input.Save(@"C:/Users/Zhigonghui/Desktop/testEmgu1/testEmgu1/imgs2/R12/out" + num.ToString() +"_"+ stats[0].ToString()+ "_origianl.png");
                //img_measured.Save("C:/Users/Zhigonghui/Desktop/testEmgu1/testEmgu1/imgs2/out/" + num.ToString() + "_origianl.png");
                //img_measured1.Save(@"C:\Users\Zhigonghui\Desktop\testEmgu1\testEmgu1\R12/out/" + num.ToString() + "_BW.png");
                //ImageViewer.Show(img_measured);
            }

            stopWatch.Stop();
            Debug.WriteLine("RunTime " + stopWatch.Elapsed);


            // USING MATLAB DLL DEMO
            /*   
            MLApp.MLApp matlab = new MLApp.MLApp();
            matlab.Execute(@"cd D:\工业视觉项目\BS翘扣头\翘扣头图像");
            object result = null;
            matlab.Feval("imread", 1, out result, "D:/工业视觉项目/BS翘扣头/翘扣头图像/NGs2/120608251400_1_2.jpg");
            var res = result as object[];
            byte[,] img_byte = (byte[,])res[0];

            /*
            int width = img.GetLength(0);
            int height = img.GetLength(1);
            int stride = width * 4;

            double[,] integers = new double[width, height];

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    integers[x, y] = img[x,y];
                }
            }
            
            //            object results_output = null;
            //            matlab.Feval("MWdetectBoudary", 2, out results_output, img);



            QKTclass mydetect = new QKTclass();
            MWArray[] resultsarrray = mydetect.MWDLLDetectQKT(2, (MWNumericArray)img_byte);
            MWNumericArray _output0 = (MWNumericArray)resultsarrray[0];
            MWNumericArray _output1 = (MWNumericArray)resultsarrray[1];
            Array byte_outputIMG = _output0.ToArray();
            Array byte_outputPoints = _output1.ToArray();

            */


        }
    }
}
