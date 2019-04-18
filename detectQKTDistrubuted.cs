using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections;
using System.IO;
using System.Diagnostics;
using Emgu.CV.UI;

namespace testDLL
{
    class detectQKTDistrubuted
    {
        public int caliX = -4;
        public int caliY = -2;
        public double partialBound = 0.5;     // dealing with top bound offset： low-> points less

        /* initail variable */
        public int DERIVATIVE_Y = 7;//3;
        public int MORP = 11;
        public int GAUSSIAN_COE = 7;
        public int DISPLAY_COUNT = 4;
        public int TQIAO = 20;//70;
        public int RANGE_Y = 250;
        public int BWA = 150;



        public Image<Gray, byte> byte2emguimage(byte[] imagebytes,int w,int h)
        {
            Image<Gray, byte> img = new Image<Gray, byte>(w, h);
            for(int c = 0; c < w; c++)
            {
                for(int r = 0; r < h; r++)
                {
                    img.Data[c, r, 0] = imagebytes[c*r];
                }
            }

            return img;
        }
        //convert bytearray to image
        public Image byteArrayToImage(byte[] byteArrayIn)
        {
            using (MemoryStream mStream = new MemoryStream(byteArrayIn))
            {
                return Image.FromStream(mStream);
            }
        }
        //convert image to bytearray
        public byte[] imgToByteArray(Image img)
        {
            using (MemoryStream mStream = new MemoryStream())
            {
                img.Save(mStream, img.RawFormat);
                return mStream.ToArray();
            }
        }
        public struct CCStatsOp
        {
            public Rectangle Rectangle;
            public int Area;
        }
        public ArrayList smoothLine(ArrayList points)
        {
            ArrayList boundSmooth = new ArrayList();
            points.Add(points[points.Count-1]);
            points.Insert(0, points[0]);
            for(int n = 1; n < points.Count-1; n++)
            {
                int[] thepoint_mid = (int[])points[n];
                int[] thepoint_right = (int[])points[n - 1];
                int[] thepoint_left = (int[])points[n + 1];

                int y_average = (thepoint_left[1] + thepoint_mid[1] + thepoint_right[1]) /3;
                int[] thepoint = new int[] { thepoint_mid[0],y_average};

                boundSmooth.Add(thepoint);
            }

            return boundSmooth;
        }
        private Image<Gray, byte> bwareaopen(Image<Gray, byte> Input_Image, int threshold)
        {

            Image<Gray, byte> bwresults = Input_Image.Copy();
            var labels = new Mat();
            var stats = new Mat();
            var centroids = new Mat();
            int nLabels = CvInvoke.ConnectedComponentsWithStats(bwresults, labels, stats, centroids);
            Image<Gray, byte> imageparts;
            imageparts = Input_Image.CopyBlank();
            var centroidPoints = new MCvPoint2D64f[nLabels];
            centroids.CopyTo(centroidPoints);
            CCStatsOp[] statsOp = new CCStatsOp[stats.Rows];
            stats.CopyTo(statsOp);
            
            foreach (var statop in statsOp)
            {
               if (statop.Area < threshold)
                {
                    //Console.WriteLine($"Rectangle: {statop.Rectangle} Area: {statop.Area}");
                    //bwresults.Draw(statop.Rectangle,new Gray(0),5);
                    for(int c = statop.Rectangle.X; c < statop.Rectangle.X+statop.Rectangle.Width; c++)
                    {
                        for(int r = statop.Rectangle.Y; r < statop.Rectangle.Y+statop.Rectangle.Height; r++)
                        {
                            bwresults.Data[r, c, 0] = 0;
                        }
                    }
                }
            }

            return bwresults;
        }
        public ArrayList DoDetection(byte[] imagebyte)
        {
            if(imagebyte == null)
            {
                Debug.WriteLine("detectQKTDistrubuted input image byte is null");
            }

            // add other stats to the results
            ArrayList result_stats = new ArrayList();

            // outlier detected result
            bool outlier_stats;

            Image ii = byteArrayToImage(imagebyte);
            Bitmap bimi = (Bitmap)ii;
            Image<Gray, Byte> img_input = new Image<Gray, Byte>(bimi);

           // ImageViewer.Show(img_input); //check the image is correctly converted from byte[]

            // output with image, bw, points
            ArrayList result_output = new ArrayList();
            Image<Gray, byte> img1 = img_input;

            Image<Gray, byte> img_normal = img1;    // for normal image bitmap
            Image<Gray, byte> img_obvious = img1;   // for obvious image bitmap

            int headoffset = 3;
            int counter_normalQKT = 0;
            int counter_obviousQKT = 0;
            int counter_CLP = 0;
            /* Canny Edge */
            Image<Gray, byte> Img_Otsu_Gray = img1.CopyBlank();
            Image<Gray, byte> Img_edge_BW = img1.CopyBlank();
            double CannyAccThresh = CvInvoke.Threshold(img1, Img_Otsu_Gray, 0, 255, Emgu.CV.CvEnum.ThresholdType.Otsu);
            //double CannyThresh = (1/2)*CannyAccThresh;

            CannyAccThresh = 0.5 * CannyAccThresh;
            double CannyThresh = 0.5 * CannyAccThresh;

            img1 = img1.SmoothGaussian(GAUSSIAN_COE);
            Img_edge_BW = img1.Canny(CannyThresh, CannyAccThresh);   // setting for Canny1
            //ImageViewer.Show(Img_edge_BW);

            /* Morphology */
            int offset_morph = MORP;
            Image<Gray, byte> Img_dilate = Img_edge_BW.CopyBlank();
            //Mat erodeKernel = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Ellipse, new Size(5,5), new Point(-1, -1));

            Mat dilateKernel = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(offset_morph, 1), new Point(-1, -1));
            //Img_dilate = Img_edge_BW.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Dilate, dilateKernel,new Point(-1, -1),1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1.0));

            Mat openKernel = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(101, 3), new Point(-1, -1));
            Img_dilate = Img_edge_BW.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Close, openKernel, new Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1.0));

            Img_dilate = bwareaopen(Img_dilate, BWA);
            //ImageViewer.Show(Img_dilate);


            int row = Img_dilate.Rows;
            int col = Img_dilate.Cols;
            
            // find the dense bound points for coordinating the corner point
            ArrayList topBound = new ArrayList();
            for (int c= 0; c<col; c=c+2)  // take half of the bound for analysis
            {
                for (int r = 100; r < row; r=r+1)
                {
                    double thispixelvalue = Img_dilate.Data[r,c,0]/255;
                    //Debug.WriteLine(r.ToString()+','+c.ToString()+','+thispixelvalue.ToString());
                    if (thispixelvalue == 1)
                    {
                        int[] thispoint = { c, r };
                        topBound.Add(thispoint);
                        break;
                    }
                }
            }

            // calculate curvature and derivative for coordinating the corner point
            ArrayList curvatures = new ArrayList();
            ArrayList normals = new ArrayList();
            ArrayList firstboudID = new ArrayList();
            ArrayList secondboudID = new ArrayList();

            ArrayList thefirst = new ArrayList();
            ArrayList theSecond = new ArrayList();

            if(topBound.Count == 0)
            {
                Debug.WriteLine("primery bound not found");
                outlier_stats = false;
                result_stats.Add(outlier_stats);
                result_output.Add(imagebyte);
                result_output.Add(imagebyte);
                result_output.Add(null);
                result_output.Add(result_stats);

                return result_output;
            }
            int[] boarder1 = (int[]) topBound[0];
            int[] boarder2 = (int[]) topBound[topBound.Count-1];
            topBound.Insert(0, boarder1);
            topBound.Insert(0, boarder1);
            topBound.Add(boarder2);
            topBound.Add(boarder2);

            // smooth the bound 
            for(int iteration = 0; iteration < 0; iteration++)
            {
                topBound = smoothLine(topBound);
            }
            

            int startnum = (int)(topBound.Count * partialBound);
            for (int num = startnum; num < topBound.Count-3; num=num+1)  // define ROI for finding the corner point
            {
                int[] thePoint = (int[]) topBound[num];

                int[] prvpoint = (int[])topBound[num - 1];
                int[] nextpoint = (int[])topBound[num + 1];
                int[] prepoint1 = (int[])topBound[num - 2];
                int[] nextpoint1 = (int[])topBound[num + 2];

                double thePoint_xd1 = nextpoint[0] - prvpoint[0];
                double thePoint_yd1 = nextpoint[1] - prvpoint[1];

                double thePoint_xd2 = nextpoint1[0] + prepoint1[0] - 2 * thePoint[0];
                double thePoint_yd2 = nextpoint1[1] + prepoint1[1] - 2 * thePoint[1];

                double k1 = Math.Abs(thePoint_xd1 * thePoint_yd2 - thePoint_yd1 * thePoint_xd2);
                double k2 = Math.Pow(Math.Sqrt(Math.Pow(thePoint_xd2,2) + Math.Pow(thePoint_yd2, 2)),3);
                double theK = k1 / k2;
                
                curvatures.Add(theK);
                // cal normal
                double dx = nextpoint[0] - thePoint[0];
                double dy = nextpoint[1] - thePoint[1];
                double[] thisnormal = {-dy, dx};
                normals.Add(thisnormal);


                double dx1 = thePoint[0] - prvpoint[0] ;
                double dy1 = thePoint[1] - prvpoint[1];
                double[] thisnormal1 = { -dy1, dx1 };

                if (thisnormal[0] * thisnormal[1] <= 0)
                {
                    firstboudID.Add(num);
                    thefirst.Add(thePoint);
                    int[] prepoint2 = (int[])topBound[num - 3];
                    int[] nextpoint2 = (int[])topBound[num + 3];
                    int diff = (int)thePoint_yd1 +(nextpoint2[1]-prepoint2[1]) + (nextpoint1[1] - prepoint1[1]);
                    //int diff = (int)thePoint_yd1 + (nextpoint1[1] - prepoint1[1]);

                    //Console.WriteLine(num.ToString() + ": " + diff.ToString() + "," + thePoint_yd2.ToString() +","+ thePoint[1]);
                    if (diff > DERIVATIVE_Y && thePoint_yd2 > 0 && thePoint[0]> col/2)
                    {
                        //Console.WriteLine("2nd: " + diff.ToString());
                        
                        secondboudID.Add(num);
                        theSecond.Add(thePoint);
                    }
                }

            }

            /* handle no firstbound , no second bound */
            int[] cornerpoint;
            if (secondboudID.Count == 0 && firstboudID.Count > 0)
            {
                counter_CLP++;
                int index = (int) firstboudID[firstboudID.Count - 1];
                int[] points = (int[])topBound[index];
                cornerpoint = new int[] { points[0], points[1] };
            }
            else if(secondboudID.Count == 0 && firstboudID.Count == 0)
            {
                int[] points = (int[])topBound[topBound.Count-1];
                cornerpoint = new int[] { points[0], points[1] };
            }
            else
            {
                if (secondboudID.Count < 300)
                {
                    int index = (int)secondboudID[0];
                    int[] points = (int[])topBound[index];
                    cornerpoint = (int[])topBound[index];
                }
                else
                {
                    int medianvar = 0;
                    for (int iter = 0; iter < secondboudID.Count; iter++)
                    {
                        int thisindex = (int)secondboudID[iter];
                        int[] thispoint = (int[])topBound[thisindex];
                        medianvar = medianvar + thispoint[0];
                    }
                    medianvar = medianvar / secondboudID.Count;

                    int minvar = 10000;
                    int minidex = 0;
                    for (int iter = 0; iter < secondboudID.Count; iter++)
                    {
                        int thisindex = (int)secondboudID[iter];
                        int[] thispoint = (int[])topBound[thisindex];

                        int diff = Math.Abs(thispoint[0] - medianvar);
                        if (diff < minvar)
                        {
                            minvar = diff;
                            minidex = thisindex;
                        }
                    }

                    cornerpoint = (int[])topBound[minidex];
                }

                
                /* find the closest point to the righttop corner
                //cornerpoint = new int[] { points[0], points[1] };
                double mindist = 1000;
                int indeofMin = 0;
                for (int iter = 0; iter < secondboudID.Count; iter++)
                {
                    int thisindex = (int)secondboudID[iter];
                    int[] thispoint = (int[])topBound[thisindex];

                    double dist_square = Math.Pow((Img_dilate.Cols - thispoint[0]),2) + Math.Pow(thispoint[1], 2);
                    double dist = Math.Sqrt(dist_square);

                    //double colDist = thispoint[1]/(Img_dilate.Cols - thispoint[0]);
                    if (dist < mindist )
                    {
                        mindist = dist;
                        indeofMin = iter;
                        
                    }
                }
                int thisMINindex = (int)secondboudID[indeofMin];
                int[] thisMINpoint = (int[])topBound[thisMINindex];
                cornerpoint = (int[])thisMINpoint.Clone();
                */
            }

            /* finding the edge accroding to the corner point */
            double CannyAccThresh1 = CvInvoke.Threshold(img1, Img_Otsu_Gray, 0, 255, Emgu.CV.CvEnum.ThresholdType.Otsu);
            CannyAccThresh1 = 0.5 * CannyAccThresh1;
            double CannyThresh1 = 0.5 * CannyAccThresh1;

            Image<Gray, byte> Img_bound_BW = img1.Canny(CannyThresh1, CannyAccThresh1);      //setting for Canny2
                                                                                             //Image<Gray, byte> Img_bound_BW = Img_dilate;
                                                                                             //ImageViewer.Show(Img_bound_BW);
            Img_bound_BW = Img_dilate;
            //ImageViewer.Show(Img_bound_BW);
            // display results and save as img file

            //Image<Gray, byte> img_input = img_input.ThresholdBinary(new Gray(70),new Gray(255));

            Bitmap bmpTif = new Bitmap(img_normal.Bitmap);

            Bitmap bmp = bmpTif.Clone(new Rectangle(0, 0, img_normal.Cols, img_normal.Rows),
                System.Drawing.Imaging.PixelFormat.Format32bppRgb);

            Graphics g = Graphics.FromImage(bmp);
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

            Image<Gray, byte> Img_BW = img1.Canny(30, 55);
            Bitmap bmpTif1 = new Bitmap(Img_dilate.Bitmap);
            Bitmap bmp1 = bmpTif1.Clone(new Rectangle(0, 0, Img_BW.Cols, Img_BW.Rows),
                System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            Graphics g1 = Graphics.FromImage(bmp1);
            g1.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

            // full bound display //////////////////////////////////test
            /* */
            //Matrix<double> mat = new Matrix<double>(topBound.Count,1);
            //for (int num = 0; num < topBound.Count; num++)
            //{
            //    int[] thepoint = (int[])topBound[num];
            //    mat[num, 0] = (double)thepoint[1];
            //}


            //Matrix<double> outmat = new Matrix<double>(topBound.Count, 1);
            //CvInvoke.GaussianBlur(mat, outmat, new Size(1, 1),1, 1);

            for (int num = 0; num < topBound.Count - 2; num++)
            {
                int OFFSETRECALI = 15;
                int[] thepoint = (int[])topBound[num];
                g1.DrawString(".", new Font("Tahoma", 15, FontStyle.Regular), Brushes.Blue, new Point(thepoint[0]-15, thepoint[1]-15));
            }

            for (int num = 0; num < thefirst.Count; num++)
            {
                int[] thepoint = (int[])thefirst[num];
                int OFFSETRECALI = 15;
                g1.DrawString("+ ", new Font("Tahoma", OFFSETRECALI), Brushes.Red, new Point(thepoint[0]-15, thepoint[1]-15 ));
                //g1.DrawString(num.ToString(), new Font("Tahoma", 10), Brushes.Red, new Point(thepoint[0], thepoint[1] - OFFSETRECALI-10-num*2));
            }

            for (int num = 0; num < theSecond.Count; num++)
            {
                int[] thepoint = (int[])theSecond[num];
                int OFFSETRECALI = 15;
                g1.DrawString("+ ", new Font("Tahoma", OFFSETRECALI), Brushes.Green, new Point(thepoint[0]-15, thepoint[1]-15 ));
                //g1.DrawString(num.ToString(), new Font("Tahoma", 10), Brushes.Green, new Point(thepoint[0], thepoint[1] - OFFSETRECALI-10));
            }

            Random rnd = new Random();
           
            //bmp1.Save(@"D:\工业视觉项目\BS_翘扣头\_QKT\testEmgu1\testEmgu1\TestIMG\R13\out/" + rnd.NextDouble().ToString() + ".JPG");
            ImageViewer.Show(new Image<Bgr, byte>(bmp1));
            

            int row1 = Img_bound_BW.Rows;
            int col1 = Img_bound_BW.Cols;
            int X = cornerpoint[0] - 3;
            int Y = cornerpoint[1] + headoffset;
            int counter = 0;
            ArrayList bound = new ArrayList();
            bound.Add(new int[] { X,Y});
            for (int c = 20; c < X-1; c=c+20)
            {
                int column = X - c;
                bool isFound = false;
                int Yrange = Y + RANGE_Y;
                if (Yrange > 580)
                {
                    Yrange = 580;
                }
                for (int r = Y-50; r < Yrange; r++)    
                {
                    double thispixelvalue = Img_bound_BW.Data[r, column, 0] / 255;

                    if (thispixelvalue == 1)
                    {
                        int[] thispoint = { column, r };
                        bound.Add(thispoint);
                        for (int iter =1; iter <= counter; iter++)
                        {
                            int offset = r - Y;
                            int[] newvar = { column+20*iter, r - ((offset * (iter)) / (counter + 1))};
                            bound.Insert(bound.Count - iter-1, newvar);
                            bound.RemoveAt(bound.Count - iter-1);
                        }
                        isFound = true;
                        counter = 0;

                        break;
                    }
                }
                if (!isFound)
                {
                    counter++;
                    int[] lastRow = (int[])bound[bound.Count - 1];
                    int[] newvar = { column, (lastRow[1]+Y)/2 - counter };
                    bound.Add(newvar);
                }
            }




            /* smoothing*/
            double threshCorrect = 10;
            ArrayList dyy = new ArrayList();
            ArrayList bound_first = new ArrayList();  // right -> left
            ArrayList bound_smooth = new ArrayList();  // left -> right
            ArrayList bound_board = new ArrayList();  // left -> right
            bound_board = (ArrayList)bound.Clone();
            bound_board.Add(bound[bound.Count-1]);
            bound_board.Insert(0, bound[0]);
            for(int num = 1; num < bound_board.Count-1; num++)
            {
                int[] thepoint_mid = (int[])bound_board[num];
                int[] thepoint_right = (int[])bound_board[num-1];
                int[] thepoint_left = (int[])bound_board[num+1];

                double thePoint_yd1 = thepoint_right[1] + thepoint_left[1] - 2 * thepoint_mid[1];
                dyy.Add(thePoint_yd1);

                int[] thepoint = new int[] { thepoint_mid[0], (thepoint_left[1] + thepoint_mid[1] + thepoint_right[1]) / 3 };

                bound_smooth.Add(thepoint);
                bound_first.Add(thepoint);
            }


            
            bound_board.Reverse();
            dyy.Reverse();

            /* correction  */
            int correctnum = 0;
            for (int num = 0; num < dyy.Count ; num++)
            {
                int[] thepoint = (int[])bound_board[num+1];
                int[] thepoint_prv = (int[])bound_board[num ];
                int[] thepoint_next = (int[])bound_board[num +2];

                double L2;
                if (num ==0)
                {
                    L2 = thepoint_prv[1];
                }
                else
                {
                    int[] thepoint_prv2 = (int[])bound_board[num-1];
                    L2 = thepoint_prv2[1];
                }


                if (Math.Abs(Convert.ToDouble(dyy[num])) > 10)
                {
                    correctnum++;
                    bound_board[num + 1] = new int[] { thepoint[0], (int)(thepoint_prv[1] * 2 - L2) };
                }
                else
                {
                    if(correctnum > 15)
                    {
                        break;
                    }
                    for (int i = 0; i < correctnum; i++)
                    {
                        double C;
                        int[] head =(int[]) bound_board[num + 1];
                        int[] tail = (int[]) bound_board[num + 1 - (correctnum + 1)];
                        if (i == 0)
                        {
                            C = 0.1;
                        }
                        else
                        {
                            C = 0;
                        }
                        if (head[1] - tail[1] > 0)
                        {
                            bound_board[num + 1 - (i + 1)] = new int[] { thepoint[0]-20*(i+1),(int) (thepoint[1]- (head[1] - tail[1])*((i+C)/correctnum)) };
                        }
                        else if(head[1] == tail[1])
                        {
                            bound_board[num + 1 - (i + 1)] = new int[] { thepoint[0]- 20 * (i + 1), thepoint[1] };
                        }
                        else
                        {
                            bound_board[num + 1 - (i + 1)] = new int[] { thepoint[0]- 20 * (i + 1), (int)(thepoint[1] - (head[1] - tail[1]) * ((i+C)/ correctnum)) };
                        }
                        
                    }
                    correctnum = 0;
                }

            }
            
            
            bound_smooth = smoothLine(bound_board);
           
            /* segmentation  */
            //Mat locpoints = new Mat(bound.Count,2,Emgu.CV.CvEnum.DepthType.Cv64F,1);
            ArrayList clusterList = new ArrayList();
            double thresholdSSD = 10;
            for (int num1 = 0; num1 < bound.Count; num1++)
            {
                ArrayList cluster = new ArrayList();
                int[] thepoint1 = (int[])bound[num1];
                Point drawpoint1 = new Point(thepoint1[0] + caliX, thepoint1[1] + caliY);

                for(int num2 = 0; num2 < bound.Count; num2++)
                {
                    int[] thepoint2 = (int[])bound[num2];
                    Point drawpoint2 = new Point(thepoint2[0] + caliX, thepoint2[1] + caliY);

                    double ssd = Math.Abs(drawpoint1.Y - drawpoint2.Y);

                    if (ssd < thresholdSSD)
                    {
                        cluster.Add(drawpoint2);
                        bound.RemoveAt(num2);
                        num2--;
                    }
                }
                num1--;
                clusterList.Add(cluster);
            }




            // finally add to result list
            int offsetCounter = 0;
            ArrayList result_bound = new ArrayList();
            bound_smooth.Reverse();
            dyy.Reverse();
            for (int num = 0; num < dyy.Count-DISPLAY_COUNT; num++)
            {
                int[] thepoint = (int[])bound_smooth[num+1];
                int OFFSETRECALI = 15;
                Point drawpoint = new Point(thepoint[0] + caliX, thepoint[1] + caliY);

                double thek = Convert.ToDouble(dyy[num] );

                /**/
                if (Math.Abs(thek) > threshCorrect)
                {
                    g.DrawString("x", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));
                    g1.DrawString("x", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));

                }
                else
                {
                    g.DrawString("+", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));
                    g1.DrawString("+", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));

                }

                //ImageViewer.Show(new Image<Bgr, byte>(bmp1));

                //g1.DrawString("+", new Font("Tahoma", 1, FontStyle.Bold), Brushes.Red, drawpoint);  // test exact points
                result_bound.Add(drawpoint);

                // bound offset rate
                for(int r = 50; r < thepoint[1]-10; r++)
                {
                    int thepixel = Img_BW.Data[r, thepoint[0],0] /255;
                    if(thepixel == 1)
                    {
                        offsetCounter++;
                        break;
                    }
                }

            }
            string boundOffset = offsetCounter.ToString()+" / "+bound_smooth.Count.ToString();
            //Debug.WriteLine(boundOffset);

            // anaylse clusters
            ArrayList clusterMeans = new ArrayList();
            
            for(int num = 0; num < clusterList.Count; num++)
            {
                double Ymean = 0;
                ArrayList theCluster =(ArrayList) clusterList[num];
                for(int c=0; c < theCluster.Count; c++)
                {
                    int OFFSETRECALI = 15;
                    Point drawpoint = (Point)theCluster[c];
                    //g.DrawString(num.ToString(), new Font("Tahoma", 15, FontStyle.Italic), Brushes.White, new Point(drawpoint.X - OFFSETRECALI, drawpoint.Y - OFFSETRECALI*2));
                    //g1.DrawString(num.ToString(), new Font("Tahoma", 15, FontStyle.Italic), Brushes.White, new Point(drawpoint.X - OFFSETRECALI, drawpoint.Y - OFFSETRECALI*2));

                    Ymean = Ymean + drawpoint.Y;
                }
                double Ymean_all = Ymean / theCluster.Count;
                clusterMeans.Add(Ymean_all);
            }

            int outlierCount = 0;
            double sumDiff = 0;
            double thresholdOutlier = 1;
            for(int n = 0; n < clusterMeans.Count; n++)
            {
                if(clusterMeans.Count == 1)
                {
                    outlierCount = 0;
                    sumDiff = 0;
                }else if(clusterMeans.Count == 2)
                {
                    if(n < clusterMeans.Count - 1)
                    {
                        double n1 = (double)clusterMeans[n];
                        double n2 = (double)clusterMeans[n + 1];
                        double d1 = Math.Abs(n1 - n2);

                        sumDiff = sumDiff + d1;
                        if (d1 > thresholdOutlier)
                        {
                            outlierCount++;
                        }
                    }
                    
                }
                else
                {

                    double n1 = 0;
                    double n2 = 0;
                    double n3=0;
                    if (n< clusterMeans.Count - 2)
                    {
                         n1 = (double)clusterMeans[n];
                         n2 = (double)clusterMeans[n + 1];
                         n3 = (double)clusterMeans[n + 2];
                    }else if(n < clusterMeans.Count - 1)
                    {
                         n1 = (double)clusterMeans[n];
                         n2 = (double)clusterMeans[n + 1];
                         n3 = (double)clusterMeans[n + 1];
                    }else if (n < clusterMeans.Count)
                    {
                         n1 = (double)clusterMeans[n];
                         n2 = (double)clusterMeans[n];
                         n3 = (double)clusterMeans[n];
                    }
                    

                    double d1 = Math.Abs(n1 - n2);
                    double d2 = Math.Abs(n1 - n3);

                    sumDiff = sumDiff + d1;
                    if (n1 - n2 > 0)
                    {
                        outlierCount++;
                    }
                }
                

            }

            // show on image 
            /*
            g1.DrawString("翘扣性质: 普通翘扣", new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 20));
            g1.DrawString("簇: "+clusterList.Count.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20,40));
            g1.DrawString("异常因子数: " + outlierCount.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 60));
            g1.DrawString("累加差值: " + sumDiff.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 80));
            g1.DrawString("边界偏移比例: " + boundOffset.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 100));

            */
            int outlierThresh = 3;
            int accdiffThresh = 100;


            double boundoffsetscaler = (double)offsetCounter / (double)bound_smooth.Count;
            if (sumDiff > accdiffThresh || boundoffsetscaler > 0.2)
            {
                outlier_stats = false;
                //g1.DrawString("x 图像异常 x", new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 130));
            }
            else
            {
                outlier_stats = true;
                //g1.DrawString("- 图像正常 -", new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 130));
            }
            

            for (int num = 0;num< Img_bound_BW.Rows; num=num+1)
            {
                if ((num%5) == 0)
                {
                    Point drawpoint = new Point(0, num);
                    g1.DrawString("-", new Font("Tahoma", 10), Brushes.White, drawpoint);
                }
                if((num % 20) == 0)
                {
                    Point drawpoint = new Point(0, num);
                    g1.DrawString("—", new Font("Tahoma", 10), Brushes.White, drawpoint);
                }
            }


            // extral process for obvious and normal

            ArrayList fullbound = topBound;
            //int Tqiao = 70;                     //condition for extral process when comparing fist point and last point
            int numberOfconcern = 2;

            int[] a = (int[])bound_first[0];
            int[] b = (int[])bound_first[1];
            double cornerpoints = (a[1] + b[1]) / numberOfconcern;

            int[] z0,z1, z2, z3, z4;
            double endpoints;
            if (fullbound.Count > 100)
            {
                z0 = (int[])fullbound[20];
                z1 = (int[])fullbound[40];
                z2 = (int[])fullbound[60];
                z3 = (int[])fullbound[80];
                z4 = (int[])fullbound[100];
                endpoints = (z0[1] + z1[1] + z2[1] + z3[1] + z4[1])/5;
            }
            else
            {
                endpoints = 0;
                Debug.WriteLine("fullbound less than 5 points");
            }
            


            
            double isQiao = endpoints - cornerpoints;
            Debug.WriteLine("Do reQKT: " + isQiao.ToString());
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            if (isQiao > TQIAO)
            {
                
                ArrayList result_boundrecal = new ArrayList();
                int[] initialpoint;
                int initalIndex = 0;


                /* take topest first bound */
                if (firstboudID.Count > 0 && isQiao>59 && topBound.Count > 40)
                {
                    int index = (int)firstboudID[0];
                    initialpoint = (int[])topBound[index];
                    int acc = 0;
                    for (int iter = 0; iter < firstboudID.Count; iter++)
                    {
                        int thisindex = (int)firstboudID[iter];
                        int[] thispoint = (int[])topBound[thisindex];

                        if (thispoint[1] < initialpoint[1])
                        {
                            initialpoint = (int[])thispoint.Clone();
                            initalIndex = thisindex;

                            int[] thispoint1;
                            int[] thispoint2;
                            int[] thispoint3;
                            int[] thispoint4;
                            if (topBound.Count > thisindex + 20)
                            {
                                thispoint1 = (int[])topBound[thisindex - 10];
                                thispoint2 = (int[])topBound[thisindex + 10];
                                thispoint3 = (int[])topBound[thisindex - 20];
                                thispoint4 = (int[])topBound[thisindex + 20];
                            }
                            else if (topBound.Count > thisindex + 10)
                            {
                                thispoint1 = (int[])topBound[thisindex - 10];
                                thispoint2 = (int[])topBound[thisindex + 10];
                                thispoint3 = (int[])topBound[thisindex - 20];
                                thispoint4 = (int[])topBound[topBound.Count - 1];
                            }
                            else
                            {
                                thispoint1 = (int[])topBound[thisindex - 10];
                                thispoint2 = (int[])topBound[topBound.Count - 1];
                                thispoint3 = (int[])topBound[thisindex - 20];
                                thispoint4 = (int[])topBound[topBound.Count - 1];
                            }
                            acc = Math.Abs(thispoint4[1] - thispoint2[1]) + Math.Abs(thispoint2[1] - thispoint[1]) + Math.Abs(thispoint[1] - thispoint1[1])+ Math.Abs(thispoint1[1] - thispoint3[1]);
                        }
                    }
                    if (acc < 10)
                    {

                        initialpoint = (int[])topBound[topBound.Count - 1];

                        for (int iter = 0; iter < secondboudID.Count; iter++)
                        {
                            int thisindex = (int)secondboudID[iter];
                            int[] thispoint = (int[])topBound[thisindex];

                            if (thispoint[1] < initialpoint[1])
                            {
                                initialpoint = (int[])thispoint.Clone();
                                initalIndex = thisindex;
                            }
                        }
                    }
                }
                else if (isQiao < 70 && isQiao>TQIAO && secondboudID.Count>0 && topBound.Count>40)
                {
                    int index = (int)firstboudID[0];
                    initialpoint = (int[])topBound[index];
                    int acc = 0;
                    for (int iter = 0; iter < firstboudID.Count; iter++)
                    {
                        int thisindex = (int)firstboudID[iter];
                        int[] thispoint = (int[])topBound[thisindex];

                        if (thispoint[1] < initialpoint[1])
                        {
                            initialpoint = (int[])thispoint.Clone();
                            initalIndex = thisindex;

                            int[] thispoint1;
                            int[] thispoint2;
                            int[] thispoint3;
                            int[] thispoint4;
                            if (topBound.Count> thisindex + 20)
                            {
                                thispoint1 = (int[])topBound[thisindex - 10];
                                thispoint2 = (int[])topBound[thisindex + 10];
                                thispoint3 = (int[])topBound[thisindex - 20];
                                thispoint4 = (int[])topBound[thisindex + 20];
                            }
                            else if(topBound.Count > thisindex + 10)
                            {
                                thispoint1 = (int[])topBound[thisindex - 10];
                                thispoint2 = (int[])topBound[thisindex + 10];
                                thispoint3 = (int[])topBound[thisindex - 20];
                                thispoint4 = (int[])topBound[topBound.Count-1];
                            }
                            else
                            {
                                thispoint1 = (int[])topBound[thisindex - 10];
                                thispoint2 = (int[])topBound[topBound.Count-1];
                                thispoint3 = (int[])topBound[thisindex - 20];
                                thispoint4 = (int[])topBound[topBound.Count-1];
                            }
                            
                            acc = Math.Abs(thispoint4[1] - thispoint2[1]) + Math.Abs(thispoint2[1] - thispoint[1]) + Math.Abs(thispoint[1] - thispoint1[1]) + Math.Abs(thispoint1[1] - thispoint3[1]);
                        }
                    }
                    if (acc < 10)
                    {
                        initialpoint = (int[])topBound[topBound.Count - 1];

                        for (int iter = 0; iter < secondboudID.Count; iter++)
                        {
                            int thisindex = (int)secondboudID[iter];
                            int[] thispoint = (int[])topBound[thisindex];

                            if (thispoint[1] < initialpoint[1])
                            {
                                initialpoint = (int[])thispoint.Clone();
                                initalIndex = thisindex;
                            }
                        }
                    }
                    
                }
                else if (secondboudID.Count == 0 && firstboudID.Count>0)
                {
                    int thisindex  = (int)firstboudID[firstboudID.Count-1];
                    initialpoint = (int[])topBound[thisindex];
                }
                else
                {
                    initialpoint = (int[])topBound[topBound.Count - 1];

                    for (int iter = 0; iter < topBound.Count; iter++)
                    {
                        int[] thispoint = (int[])topBound[iter];

                        if (thispoint[1] < initialpoint[1])
                        {
                            initialpoint = (int[])thispoint.Clone();
                            initalIndex = iter;
                        }
                    }
                }





                /* take topest second bound 
                if (firstboudID.Count >0 && secondboudID.Count>0)
                {
                    int index = (int)secondboudID[0];
                    initialpoint = (int[])topBound[index];
                    for (int iter = 0; iter < secondboudID.Count; iter++)
                    {
                        int thisindex = (int)secondboudID[iter];
                        int[] thispoint = (int[])topBound[thisindex];

                        if (thispoint[1] < initialpoint[1])
                        {
                            initialpoint = (int[])thispoint.Clone();
                        }
                    }

                }
                else if(firstboudID.Count>0 && secondboudID.Count== 0)
                {
                    int index = (int)firstboudID[0];
                    initialpoint = (int[])topBound[index];
                    for (int iter = 0; iter < firstboudID.Count; iter++)
                    {
                        int thisindex = (int)firstboudID[iter];
                        int[] thispoint = (int[])topBound[thisindex];

                        if (thispoint[1] < initialpoint[1])
                        {
                            initialpoint = (int[])thispoint.Clone();
                        }
                    }
                }
                else
                {
                    initialpoint = (int[])topBound[topBound.Count-1];

                    for (int iter = 0; iter < topBound.Count; iter++)
                    {
                        int[] thispoint = (int[])topBound[iter];

                        if (thispoint[1] < initialpoint[1])
                        {
                            initialpoint = (int[])thispoint.Clone();
                        }
                    }
                }
                */



                // when it is qiao, the smallest corner point is the corner
                int[] cornerpoint_Recal = initialpoint;
                int X_recal = cornerpoint_Recal[0] - 3;
                int Y_recal = cornerpoint_Recal[1]+ headoffset;
                //int X_recal = X;
                //int Y_recal = Y;
                //g1.DrawString("+", new Font("Tahoma", 15, FontStyle.Bold), Brushes.Yellow, new Point(X_recal,Y_recal));
                //ImageViewer.Show(new Image<Bgr, byte>(bmp1));
                int counter_recal = 0;
                ArrayList bound_recal = new ArrayList();
                bound_recal.Add(new int[] { X_recal, Y_recal });
                for (int c = 20; c < X_recal - 1; c = c + 20)
                {
                    int column = X_recal - c;
                    bool isFound = false;
                    for (int r = 100; r < row; r++)
                    {
                        double thispixelvalue = Img_bound_BW.Data[r, column, 0] / 255;

                        if (thispixelvalue == 1)
                        {
                            int[] thispoint = { column, r };
                            bound_recal.Add(thispoint);
                            for (int iter = 0; iter < counter_recal; iter++)
                            {
                                int offset = r - Y_recal;
                                int[] newvar = { column + 20 * iter, r - ((offset * (iter)) / (counter + 1)) };
                                bound_recal.Insert(bound_recal.Count - iter - 1, newvar);
                                bound_recal.RemoveAt(bound_recal.Count - iter - 1);
                            }
                            isFound = true;
                            counter_recal = 0;

                            break;
                        }
                    }
                    if (!isFound)
                    {
                        counter++;
                        int[] lastRow = (int[])bound_recal[bound_recal.Count - 1];
                        int[] newvar = { column, lastRow[1] };
                        bound_recal.Add(newvar);
                    }
                }
                ArrayList bound_seg = new ArrayList(bound_recal);
                /* smoothing */
                ArrayList dyy_recal = new ArrayList();

                ArrayList bound_smooth_recal = new ArrayList();
                ArrayList bound_board_recal = new ArrayList();
                bound_board_recal = (ArrayList)bound_recal.Clone();
                bound_board_recal.Add(bound_recal[bound_recal.Count - 1]);
                bound_board_recal.Insert(0, bound_recal[0]);
                for (int num = 1; num < bound_board_recal.Count - 1; num++)
                {
                    int[] thepoint_mid = (int[])bound_board_recal[num];
                    int[] thepoint_right = (int[])bound_board_recal[num - 1];
                    int[] thepoint_left = (int[])bound_board_recal[num + 1];

                    double thePoint_yd1 = thepoint_right[1] + thepoint_left[1] - 2 * thepoint_mid[1];
                    dyy_recal.Add(thePoint_yd1);

                    //int[] thepoint = new int[] { thepoint_mid[0], (thepoint_left[1] + thepoint_mid[1] + thepoint_right[1]) / 3 };
                    int[] thepoint = thepoint_mid;

                    bound_smooth_recal.Add(thepoint);
                }

                /* correction */
                bound_board_recal.Reverse();
                dyy_recal.Reverse();
                for (int num = 0; num < dyy_recal.Count; num++)
                {
                    int[] thepoint = (int[])bound_board_recal[num+1];
                    int[] thepoint_prv = (int[])bound_board_recal[num];
                    int[] thepoint_next = (int[])bound_board_recal[num + 2];

                    double L2;
                    if (num == 0)
                    {
                        L2 = thepoint_prv[1];
                    }
                    else
                    {
                        int[] thepoint_prv2 = (int[])bound_board_recal[num - 1];
                        L2 = thepoint_prv2[1];
                    }

                    if (Math.Abs(Convert.ToDouble(dyy_recal[num])) > threshCorrect)
                    {
                        bound_board_recal[num+1] = new int[] { thepoint[0], (int)(thepoint_prv[1] * 2 - L2) };
                    }

                }


                bound_smooth_recal = smoothLine( bound_board_recal);

                /* segmentation  */
                //Mat locpoints = new Mat(bound.Count,2,Emgu.CV.CvEnum.DepthType.Cv64F,1);
                ArrayList clusterList_recal = new ArrayList();
                double thresholdSSD_recal = 10;
                for (int num1 = 0; num1 < bound_seg.Count; num1++)
                {
                    ArrayList cluster = new ArrayList();
                    int[] thepoint1 = (int[])bound_seg[num1];
                    Point drawpoint1 = new Point(thepoint1[0] + caliX, thepoint1[1] + caliY);

                    for (int num2 = 0; num2 < bound_seg.Count; num2++)
                    {
                        int[] thepoint2 = (int[])bound_seg[num2];
                        Point drawpoint2 = new Point(thepoint2[0] + caliX, thepoint2[1] + caliY);

                        double ssd = Math.Abs(drawpoint1.Y - drawpoint2.Y);

                        if (ssd < thresholdSSD_recal)
                        {
                            cluster.Add(drawpoint2);
                            bound_seg.RemoveAt(num2);
                            num2--;
                        }
                    }
                    num1--;
                    clusterList_recal.Add(cluster);
                }




                // display results and save as img file
                
                Bitmap bmpTif_recal = new Bitmap(img_obvious.Bitmap);

                Bitmap bmp_real = bmpTif_recal.Clone(new Rectangle(0, 0, img_obvious.Cols, img_obvious.Rows),
                    System.Drawing.Imaging.PixelFormat.Format32bppRgb);

                
                Bitmap bmpTif_recal1 = new Bitmap(Img_dilate.Bitmap); //Img_edge_BW 
                //ImageViewer.Show(Img_BW);
                Bitmap bmp_real1 = bmpTif_recal1.Clone(new Rectangle(0, 0, Img_BW.Cols, Img_BW.Rows),
                    System.Drawing.Imaging.PixelFormat.Format32bppRgb);
                Graphics g_recal1 = Graphics.FromImage(bmp_real1);


                Graphics g_recal = Graphics.FromImage(bmp_real);
                g_recal.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

                // final result
                bound_smooth_recal.Reverse();
                dyy_recal.Reverse();
                int offsetCounter_recal = 0;
                for (int num = 0; num < dyy_recal.Count-DISPLAY_COUNT; num++)
                {
                    int[] thepoint = (int[])bound_smooth_recal[num+1];
                    int OFFSETRECALI = 15;
                    Point drawpoint = new Point(thepoint[0]+ caliX, thepoint[1]+ caliY);

                    double thek = Convert.ToDouble(dyy_recal[num]);
                    if(Math.Abs(thek) > threshCorrect)
                    {
                        g_recal.DrawString("o", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0], thepoint[1] - OFFSETRECALI));
                        //g_recal.DrawString("o", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0], thepoint[1] - OFFSETRECALI));
                        g_recal1.DrawString("o", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0], thepoint[1] - OFFSETRECALI));

                    }
                    else
                    {
                        g_recal.DrawString("+", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0], thepoint[1] - OFFSETRECALI));
                        g_recal1.DrawString("+", new Font("Tahoma", OFFSETRECALI, FontStyle.Bold), Brushes.White, new Point(thepoint[0], thepoint[1] - OFFSETRECALI));

                    }

                    // g_recal1.DrawString("+", new Font("Tahoma", 1, FontStyle.Bold), Brushes.Red, drawpoint);   // test exact points

                    result_boundrecal.Add(drawpoint);

                    // bound offset rate
                    for (int r = 50; r < thepoint[1] - 10; r++)
                    {
                        int thepixel;
                        if (thepoint[1] > 580)
                        {
                            thepixel = 0;
                        }
                        else
                        {
                            thepixel = Img_BW.Data[r, thepoint[0], 0] / 255;
                        }
                        
                        if (thepixel == 1)
                        {
                            offsetCounter_recal++;
                            break;
                        }
                    }
                }
                string boudoffset_recal = offsetCounter_recal.ToString() + " / " + bound_smooth_recal.Count.ToString();
                // drawing clusters
                for (int num = 0; num < clusterList_recal.Count; num++)
                {
                    ArrayList theCluster = (ArrayList)clusterList_recal[num];
                    for (int c = 0; c < theCluster.Count; c++)
                    {
                        int OFFSETRECALI = 15;
                        Point drawpoint = (Point)theCluster[c];
                        //g_recal.DrawString(num.ToString(), new Font("Tahoma", 15, FontStyle.Italic), Brushes.White, new Point(drawpoint.X - OFFSETRECALI, drawpoint.Y - OFFSETRECALI * 2));
                        //g_recal1.DrawString(num.ToString(), new Font("Tahoma", 15, FontStyle.Italic), Brushes.White, new Point(drawpoint.X - OFFSETRECALI, drawpoint.Y - OFFSETRECALI * 2));
                    }

                }
                // anaylse clusters
                ArrayList clusterMeans_recal = new ArrayList();

                for (int num = 0; num < clusterList_recal.Count; num++)
                {
                    double Ymean = 0;
                    ArrayList theCluster = (ArrayList)clusterList_recal[num];
                    for (int c = 0; c < theCluster.Count; c++)
                    {
                        int OFFSETRECALI = 15;
                        Point drawpoint = (Point)theCluster[c];
                        //g_recal.DrawString(num.ToString(), new Font("Tahoma", 15, FontStyle.Italic), Brushes.White, new Point(drawpoint.X - OFFSETRECALI, drawpoint.Y - OFFSETRECALI * 2));
                        //g_recal1.DrawString(num.ToString(), new Font("Tahoma", 15, FontStyle.Italic), Brushes.White, new Point(drawpoint.X - OFFSETRECALI, drawpoint.Y - OFFSETRECALI * 2));

                        Ymean = Ymean + drawpoint.Y;
                    }
                    double Ymean_all = Ymean / theCluster.Count;
                    clusterMeans_recal.Add(Ymean_all);
                }

                int outlierCount_recal = 0;
                double sumDiff_recal = 0;
                double thresholdOutlier_recal = 1;
                for (int n = 0; n < clusterMeans_recal.Count; n++)
                {
                    if (clusterMeans_recal.Count == 1)
                    {
                        outlierCount_recal = 0;
                        sumDiff_recal = 0;
                    }
                    else if (clusterMeans_recal.Count == 2)
                    {
                        if (n < clusterMeans_recal.Count - 1)
                        {
                            double n1 = (double)clusterMeans_recal[n];
                            double n2 = (double)clusterMeans_recal[n + 1];
                            double d1 = Math.Abs(n1 - n2);

                            sumDiff_recal = sumDiff_recal + d1;
                            if (d1 > thresholdOutlier_recal)
                            {
                                outlierCount_recal++;
                            }
                        }

                    }
                    else
                    {

                        double n1 = 0;
                        double n2 = 0;
                        double n3 = 0;
                        if (n < clusterMeans_recal.Count - 2)
                        {
                            n1 = (double)clusterMeans_recal[n];
                            n2 = (double)clusterMeans_recal[n + 1];
                            n3 = (double)clusterMeans_recal[n + 2];
                        }
                        else if (n < clusterMeans_recal.Count - 1)
                        {
                            n1 = (double)clusterMeans_recal[n];
                            n2 = (double)clusterMeans_recal[n + 1];
                            n3 = (double)clusterMeans_recal[n + 1];
                        }
                        else if (n < clusterMeans_recal.Count)
                        {
                            n1 = (double)clusterMeans_recal[n];
                            n2 = (double)clusterMeans_recal[n];
                            n3 = (double)clusterMeans_recal[n];
                        }


                        double d1 = Math.Abs(n1 - n2);
                        double d2 = Math.Abs(n1 - n3);

                        sumDiff_recal = sumDiff_recal + d1;
                        if (n1 - n2 > 0)
                        {
                            outlierCount_recal++;
                        }
                    }


                }
                /*
                g_recal1.DrawString("翘扣性质: 明显翘扣 ("+isQiao.ToString()+")", new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 20));
                g_recal1.DrawString("簇: " + clusterList_recal.Count.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 40));
                g_recal1.DrawString("异常因子数: " + outlierCount_recal.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 60));
                g_recal1.DrawString("累加差值: " + sumDiff_recal.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 80));
                g_recal1.DrawString("边界偏移比例: " + boudoffset_recal.ToString(), new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 100));

                */
                double boudoffsetscaler = (double)offsetCounter_recal / (double)bound_recal.Count;
                if (outlierCount_recal > outlierThresh || boudoffsetscaler > 0.2)
                {
                    outlier_stats = false;
                    //g_recal1.DrawString("x 图像异常 x", new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 130));
                }
                else
                {
                    outlier_stats = true;
                    //g_recal1.DrawString("- 图像正常 -", new Font("Tahoma", 15, FontStyle.Regular), Brushes.White, new Point(20, 130));
                }
                //
                for (int num = 0; num < Img_bound_BW.Rows; num = num + 1)
                {
                    if ((num % 5) == 0)
                    {
                        Point drawpoint = new Point(0, num);
                        g_recal1.DrawString("-", new Font("Tahoma", 10), Brushes.White, drawpoint);
                    }
                    if ((num % 20) == 0)
                    {
                        Point drawpoint = new Point(0, num);
                        g_recal1.DrawString("—", new Font("Tahoma", 10), Brushes.White, drawpoint);
                    }
                }

                Image<Gray, byte> result_obv = new Image<Gray, byte>(bmp_real);
                Image<Gray, byte> result_obv1 = new Image<Gray, byte>(bmp_real1);

                /* export with Image<Gray, byte> */
                //result_output.Add(result_obv);
                //result_output.Add(result_obv1);

                ImageConverter converter1 = new ImageConverter();
                byte[] result_obv_byte = (byte[])converter1.ConvertTo(result_obv.ToBitmap(), typeof(byte[]));
                byte[] result_obv_byte1 = (byte[])converter1.ConvertTo(result_obv1.ToBitmap(), typeof(byte[]));
                
                /* export with byte[] */
                result_output.Add(result_obv_byte);
                result_output.Add(result_obv_byte1);
                //ImageViewer.Show(result_obv);

                //bmp_real.Save("C:/Users/Zhigonghui/source/repos/testDLL/testDLL/imgs/2.JPG");

                /* export point location */
                result_output.Add(result_boundrecal);

                counter_obviousQKT++;

            }
            else
            {

                Image<Gray, byte> result_normal = new Image<Gray, byte>(bmp);
                Image<Gray, byte> result_normal1 = new Image<Gray, byte>(bmp1);

                /* export with Image<Gray, byte> */
                //result_output.Add(result_normal);
                //result_output.Add(result_normal1);

                ImageConverter converter2 = new ImageConverter();
                byte[] result_nor_byte = (byte[])converter2.ConvertTo(result_normal.ToBitmap(), typeof(byte[]));
                byte[] result_nor_byte1 = (byte[])converter2.ConvertTo(result_normal1.ToBitmap(), typeof(byte[]));

                /* export with byte[] */
                result_output.Add(result_nor_byte);
                result_output.Add(result_nor_byte1);
                //ImageViewer.Show(result_normal);

                /* export point location */
                result_output.Add(result_bound);

                counter_normalQKT++;
            }

            /*
            Debug.WriteLine("Num of CLP: "+ counter_CLP.ToString());
            Debug.WriteLine("Num of Obvious: " + counter_obviousQKT.ToString());
            Debug.WriteLine("Num of Normal: " + counter_normalQKT.ToString());
            */

            // adding the result outlier stats
            result_stats.Add(outlier_stats);
            result_output.Add(result_stats);


            return result_output;
        }
    }
}
