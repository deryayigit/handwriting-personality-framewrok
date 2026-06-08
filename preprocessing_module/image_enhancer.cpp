#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

struct Image
{
    int width;
    int height;
    unsigned char* data;
};

void initImage(Image& img)
{
    img.width = 0;
    img.height = 0;
    img.data = nullptr;
}

void freeImage(Image& img)
{
    delete[] img.data;
    img.data = nullptr;
    img.width = 0;
    img.height = 0;
}

unsigned char clipValue(int value)
{
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<unsigned char>(value);
}

string readTokenSkippingComments(ifstream& file)
{
    string token;

    while (file >> token)
    {
        if (!token.empty() && token[0] == '#')
        {
            string commentLine;
            getline(file, commentLine);
            continue;
        }

        return token;
    }

    throw runtime_error("Invalid PGM header.");
}

bool readPGM(const string& path, Image& img)
{
    ifstream file(path, ios::binary);

    if (!file.is_open())
    {
        cerr << "Cannot open input file: " << path << endl;
        return false;
    }

    try
    {
        string magic = readTokenSkippingComments(file);

        if (magic != "P5")
        {
            cerr << "Only binary PGM P5 format is supported." << endl;
            return false;
        }

        img.width = stoi(readTokenSkippingComments(file));
        img.height = stoi(readTokenSkippingComments(file));

        int maxValue = stoi(readTokenSkippingComments(file));

        if (maxValue != 255)
        {
            cerr << "Only 8-bit PGM images are supported." << endl;
            return false;
        }

        file.get();

        if (img.width <= 0 || img.height <= 0)
        {
            cerr << "Invalid image size." << endl;
            return false;
        }

        int size = img.width * img.height;
        img.data = new unsigned char[size];

        file.read(reinterpret_cast<char*>(img.data), size);

        if (!file)
        {
            cerr << "Error reading image data." << endl;
            freeImage(img);
            return false;
        }
    }
    catch (const exception& e)
    {
        cerr << "PGM read error: " << e.what() << endl;
        freeImage(img);
        return false;
    }

    return true;
}

bool writePGM(const string& path, const Image& img)
{
    ofstream file(path, ios::binary);

    if (!file.is_open())
    {
        cerr << "Cannot open output file: " << path << endl;
        return false;
    }

    file << "P5\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";

    int size = img.width * img.height;
    file.write(reinterpret_cast<const char*>(img.data), size);

    return true;
}

void boxBlur3x3(const Image& input, Image& output)
{
    output.width = input.width;
    output.height = input.height;

    int size = input.width * input.height;
    output.data = new unsigned char[size];

    for (int y = 0; y < input.height; y++)
    {
        for (int x = 0; x < input.width; x++)
        {
            int sum = 0;
            int count = 0;

            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny >= 0 && ny < input.height &&
                        nx >= 0 && nx < input.width)
                    {
                        sum += input.data[ny * input.width + nx];
                        count++;
                    }
                }
            }

            output.data[y * input.width + x] =
                static_cast<unsigned char>(sum / count);
        }
    }
}

void cleanBackground(Image& img)
{
    int size = img.width * img.height;

    for (int i = 0; i < size; i++)
    {
        int p = img.data[i];

        if (p > 185)
        {
            p = min(255, p + 20);
        }
        else if (p < 80)
        {
            p = max(0, p - 5);
        }

        img.data[i] = clipValue(p);
    }
}

int getLUTIndex(int tileY, int tileX, int pixelValue, int tilesX)
{
    return ((tileY * tilesX + tileX) * 256) + pixelValue;
}

void clipAndRedistributeHistogram(int histogram[256], int clipLimit)
{
    int excess = 0;

    for (int i = 0; i < 256; i++)
    {
        if (histogram[i] > clipLimit)
        {
            excess += histogram[i] - clipLimit;
            histogram[i] = clipLimit;
        }
    }

    int redistribute = excess / 256;
    int remainder = excess % 256;

    for (int i = 0; i < 256; i++)
    {
        histogram[i] += redistribute;
    }

    for (int i = 0; i < remainder; i++)
    {
        histogram[i]++;
    }
}

void computeTileLUT(
    const unsigned char* input,
    int width,
    int height,
    int startX,
    int startY,
    int tileWidth,
    int tileHeight,
    int clipLimit,
    unsigned char* lut
)
{
    int histogram[256] = {0};

    for (int y = startY; y < startY + tileHeight && y < height; y++)
    {
        for (int x = startX; x < startX + tileWidth && x < width; x++)
        {
            int index = y * width + x;
            histogram[input[index]]++;
        }
    }

    clipAndRedistributeHistogram(histogram, clipLimit);

    int cdf[256] = {0};
    cdf[0] = histogram[0];

    for (int i = 1; i < 256; i++)
    {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    int cdfMin = 0;

    for (int i = 0; i < 256; i++)
    {
        if (cdf[i] != 0)
        {
            cdfMin = cdf[i];
            break;
        }
    }

    int tilePixelCount = tileWidth * tileHeight;
    int denominator = tilePixelCount - cdfMin;

    for (int i = 0; i < 256; i++)
    {
        if (denominator > 0)
        {
            int value = ((cdf[i] - cdfMin) * 255) / denominator;
            lut[i] = clipValue(value);
        }
        else
        {
            lut[i] = static_cast<unsigned char>(i);
        }
    }
}

unsigned char interpolateLUT(
    unsigned char pixel,
    const unsigned char* lut00,
    const unsigned char* lut10,
    const unsigned char* lut01,
    const unsigned char* lut11,
    double dx,
    double dy
)
{
    double top =
        (1.0 - dx) * lut00[pixel] +
        dx * lut10[pixel];

    double bottom =
        (1.0 - dx) * lut01[pixel] +
        dx * lut11[pixel];

    double value =
        (1.0 - dy) * top +
        dy * bottom;

    return clipValue(static_cast<int>(round(value)));
}

void applyCLAHE(
    const Image& input,
    Image& output,
    int tileSize,
    int clipLimit
)
{
    output.width = input.width;
    output.height = input.height;

    int imageSize = input.width * input.height;
    output.data = new unsigned char[imageSize];

    int tilesX = (input.width + tileSize - 1) / tileSize;
    int tilesY = (input.height + tileSize - 1) / tileSize;

    int totalLUTSize = tilesX * tilesY * 256;
    unsigned char* luts = new unsigned char[totalLUTSize];

    for (int ty = 0; ty < tilesY; ty++)
    {
        for (int tx = 0; tx < tilesX; tx++)
        {
            int startX = tx * tileSize;
            int startY = ty * tileSize;

            int currentTileWidth = min(tileSize, input.width - startX);
            int currentTileHeight = min(tileSize, input.height - startY);

            unsigned char* currentLUT =
                &luts[getLUTIndex(ty, tx, 0, tilesX)];

            computeTileLUT(
                input.data,
                input.width,
                input.height,
                startX,
                startY,
                currentTileWidth,
                currentTileHeight,
                clipLimit,
                currentLUT
            );
        }
    }

    for (int y = 0; y < input.height; y++)
    {
        for (int x = 0; x < input.width; x++)
        {
            double gx = (static_cast<double>(x) / tileSize) - 0.5;
            double gy = (static_cast<double>(y) / tileSize) - 0.5;

            int x1 = static_cast<int>(floor(gx));
            int y1 = static_cast<int>(floor(gy));

            double dx = gx - x1;
            double dy = gy - y1;

            x1 = max(0, min(x1, tilesX - 1));
            y1 = max(0, min(y1, tilesY - 1));

            int x2 = max(0, min(x1 + 1, tilesX - 1));
            int y2 = max(0, min(y1 + 1, tilesY - 1));

            const unsigned char* lut00 =
                &luts[getLUTIndex(y1, x1, 0, tilesX)];

            const unsigned char* lut10 =
                &luts[getLUTIndex(y1, x2, 0, tilesX)];

            const unsigned char* lut01 =
                &luts[getLUTIndex(y2, x1, 0, tilesX)];

            const unsigned char* lut11 =
                &luts[getLUTIndex(y2, x2, 0, tilesX)];

            int index = y * input.width + x;
            unsigned char pixel = input.data[index];

            output.data[index] = interpolateLUT(
                pixel,
                lut00,
                lut10,
                lut01,
                lut11,
                dx,
                dy
            );
        }
    }

    delete[] luts;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: image_enhancer.exe input.pgm output.pgm" << endl;
        return 1;
    }

    string inputPath = argv[1];
    string outputPath = argv[2];

    Image input;
    Image smoothed;
    Image output;

    initImage(input);
    initImage(smoothed);
    initImage(output);

    if (!readPGM(inputPath, input))
    {
        return 1;
    }

    cerr << "[INFO] Image size: " << input.width << "x" << input.height << endl;

    boxBlur3x3(input, smoothed);

    int tileSize = 96;
    int clipLimit = 4;

    applyCLAHE(
        smoothed,
        output,
        tileSize,
        clipLimit
    );

    cleanBackground(output);

    if (!writePGM(outputPath, output))
    {
        freeImage(input);
        freeImage(smoothed);
        freeImage(output);
        return 1;
    }

    freeImage(input);
    freeImage(smoothed);
    freeImage(output);

    cout << "Image enhanced and saved: " << outputPath << endl;

    return 0;
}