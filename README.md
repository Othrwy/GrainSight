# GrainSight
Dimension analyser for images of small spheroidal particles.


OVERVIEW

This script allows you upload an image of semi-spheroidal particles and will scan through, picking out the smallest and largest diameter for each grain. 
It can then find the mean/SD, generate histograms and fit distributions - all outputted in a PDF.
It can report min, max, average diameters and spheroidicity.
It can also calculate volume and mass but assuming the third dimension is =Ømin and entering a density.


PHOTO QUALITY ← IMPORTANT!

→ When taking your images ensure there is a REFERENCE in shot.
→ Ensure your grains are all in focus best you can.
→ Ensure good contrast between grains and background (devised with white grains on dark, matt background).
→ Minimise shadows.


HOW TO USE

1. Upload image
2. Calibrate - pick two points, enter dimension
3. Set limits (it will pick up noise if min is too low)
4. Click "Look for Grains"
5. Click "Report Builder" (bottom left)
6. Click "+ Add Plot" (bottom right)
7. Select plot type
8. Add distribution/mean/SD (note star denotes distribution with best fit)
9. Click "Generate PDF Report"


DEV NOTES

⚠️ Multi images may be a big buggy
⚠️ Auto Calibrate is far from optimised
