#include <easyx.h>
#include <string.h>
#include <map>
#include <sstream>
#include <time.h>
#include <conio.h>
#include <iostream>

#pragma comment( lib, "MSIMG32.LIB")

// The state of the cube
enum State
{
	EXIST,
	DESTORY
};


// 2D vector, used to represent position or size
struct Vector2
{
	float x;
	float y;
};


class Block
{
private:
	State	currentState;
	State	targetState;
	Vector2	size;
	Vector2	currentPos;
	Vector2	targetPos;
	IMAGE* img;
	IMAGE* newImg;
	float	deltaPos;			// How many positions per second
	float	deltaSize;			// How much bigger per second
	float	animationSpeed;		// Animation speed


public:
	Block(const Vector2& pos, IMAGE* img)
	{
		currentPos = pos;
		targetPos = pos;
		currentState = EXIST;
		targetState = EXIST;
		size = { 50,50 };
		this->img = this->newImg = img;

		deltaPos = 100;
		deltaSize = 40;
		animationSpeed = 10.0f;
	}


	void update(float deltaTime)
	{
		// Change the size of the cube (animation from small to large when the picture is first generated)
		if (size.x < img->getwidth())
		{
			size.x = size.y = size.x + deltaSize * deltaTime * animationSpeed / 2;
			if (size.x > img->getwidth())
			{
				size.x = size.y = (float)img->getwidth();
			}
		}

		// Update the square position
		if (currentPos.x != targetPos.x || currentPos.y != targetPos.y)
		{
			if (currentPos.x != targetPos.x)
			{
				std::cout << "update_x" << std::endl;
				int direction = (targetPos.x - currentPos.x) > 0 ? 1 : -1;
				currentPos.x += deltaPos * direction * deltaTime * animationSpeed;
				// If the current position is already close to the target position, go directly to the target position
				if (std::abs(currentPos.x - targetPos.x) < 1)
					currentPos.x = targetPos.x;
			}

			// Update the status if the current position is the same as the target position
			if (currentPos.x == targetPos.x)
				currentState = targetState;
			if (currentPos.y != targetPos.y)
			{
				std::cout << "update_y" << std::endl;
				int direction = (targetPos.y - currentPos.y) > 0 ? 1 : -1;
				currentPos.y += deltaPos * direction * deltaTime * animationSpeed;
				// If the current position is already close to the target position, go directly to the target position
				if (std::abs(currentPos.y - targetPos.y) < 1)
					currentPos.y = targetPos.y;
			}

			// Update the status if the current position is the same as the target position
			if (currentPos.y == targetPos.y)
				currentState = targetState;
		}

		if (currentPos.x == targetPos.x && currentPos.y == targetPos.y)
		{
			currentState = targetState;
			img = newImg;
		}
	}


	void draw()
	{
		TransparentBlt(GetImageHDC(NULL), int(currentPos.x + (90 - size.x) / 2), int(currentPos.y + (90 - size.y) / 2),
			(int)size.x, (int)size.y, GetImageHDC(img), 0, 0, img->getwidth(), img->getheight(), BLACK);
	}


	// Moves the cube from the current position to the target position, changing its state after the move.
	void MoveTo(const Vector2& pos, IMAGE* newImg, State state = EXIST)
	{
		targetPos = pos;
		targetState = state;
		this->newImg = newImg;

		for (int i = 0; i < 4; ++i)
		{
			std::cout << "Another loop iteration: " << i << std::endl;
			for (int j = 0; j < 500000; ++j)
			{
				float result = sqrt(j * 3.0);
			}
		}

	}


	State getState()
	{
		return currentState;
	}
};



int		map[4][4];				// 4 * 4 map
Block* blockMap[4][4];		    // Square index
int		score;					// score
int		maxScore;				// Highest score
int		currentMaxBlock;		// Current largest square
int		maxBlock;				// Historical Maximum Square
int		gameLoop;				// Game Loop
float	keyTime = 0;			// Keystroke interval
std::map<int, IMAGE> image;		// Store all digital images
bool	gameOver = false;		// Whether the game is over or not
float	overTime;				// The game does not exit the loop immediately after the game ends, but waits 0.5s for the animation to update.


// Determine whether there is a way to move, return 1 if there is, return 0 if there is not.
// Detecting the idea: if you hit a grid of 0, or if two neighbouring grids have equal numbers, return 1.
int Judge()
{
	// Lateral detection
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (map[i][j] == 0 || map[i][j] == map[i][j + 1] || map[i][j + 1] == 0)
				return 1;
			if (map[i][j] == 2048 || map[i][j + 1] == 2048)
				return 0;
		}
	}

	// Vertical inspection
	for (int i = 0; i < 4; i++)
	{
		std::cout << "search" << i << std::endl;
		for (int j = 0; j < 3; j++)
		{
			if (map[j][i] == 0 || map[j][i] == map[j + 1][i] || map[j + 1][i] == 0)
				return 1;
			if (map[j][i] == 2048 || map[j + 1][i] == 2048)
				return 0;
		}
	}

	return 0;
}


void Up()
{
	int moveFlag = 0;	// Record if a move was made
	int mergeFlag = 0;	// Record if merged

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int k = j, z;

			while (k < 4 && map[k][i] == 0)
				k++;

			z = k + 1;
			while (z < 4 && map[z][i] == 0)
				z++;

			// There are non-zero squares in the current row
			if (k < 4)
			{
				if (z < 4 && map[k][i] == map[z][i])
				{
					// Can be merged
					int value = map[k][i] + map[z][i];
					map[k][i] = 0;
					map[z][i] = 0;
					map[j][i] = value;

					// Start animation
					Block* temp = blockMap[k][i];
					blockMap[k][i] = NULL;
					blockMap[j][i] = temp;
					blockMap[j][i]->MoveTo({ 25.0f + 100 * i,225.0f + 100 * j }, &image[map[j][i]]);
					blockMap[z][i]->MoveTo({ 25.0f + 100 * i,225.0f + 100 * (j + 1) }, &image[map[z][i]], DESTORY);

					// Update scores
					score += map[j][i];
					if (score > maxScore) maxScore = score;

					// Update the cube
					if (value > currentMaxBlock)
						currentMaxBlock = value;
					if (currentMaxBlock > maxBlock)
						maxBlock = currentMaxBlock;

					mergeFlag = 1;
				}
				else
				{
					// Not mergeable
					int value = map[k][i];
					map[k][i] = 0;
					map[j][i] = value;

					// Determine if you can move
					if (k != j)
					{
						moveFlag = 1;

						// Start animation
						Block* temp = blockMap[k][i];
						blockMap[k][i] = NULL;
						blockMap[j][i] = temp;
						blockMap[j][i]->MoveTo({ 25.0f + 100 * i,225.0f + 100 * j }, &image[map[j][i]]);
					}
				}
			}
			else		// Determine the next line
			{
				break;
			}
		}
	}

	// If a move or merge occurs, randomly generate a 2 or 4
	if (moveFlag || mergeFlag)
	{
		bool index;// Guidelines for the presence or absence of spaces
		int index_x, index_y;	// Index of random positions
		index = false;//Initialise index
		for (int i = 0; i < 1000; i++)
		{
			index_x = i % 4;
			index_y = i / 4;
		}
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++) {
				if (map[i][j] == 0)
				{
					index = true;
					std::cout << "find 0 " << std::endl;
					break;
				}

			}
		}
		// Finding the space position
		index_x = rand() % 4;
		index_y = rand() % 4;
		while (index)
		{
			while (map[index_x][index_y])
			{
				index_x = rand() % 4;
				index_y = rand() % 4;
			}
			break;
		}

		// 80% generate 2 , 20% generate 4
		int num = rand() % 10;
		if (num < 8)
		{
			map[index_x][index_y] = 2;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[2]);
		}
		else
		{
			map[index_x][index_y] = 4;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[4]);
		}
	}
}



void Down()
{
	int moveFlag = 0;	// Record if a move was made
	int mergeFlag = 0;	// Record if merged

	for (int i = 0; i < 4; i++)
	{
		for (int j = 3; j > 0; j--)
		{
			int k, z;

			// Find a square that is not 0 and move it down and determine if it can be merged with the square above it.
			for (k = j; k >= 0; k--)
				if (map[k][i] != 0)
					break;

			// Look for squares with a right-hand side that is not 0
			for (z = k - 1; z >= 0; z--)
				if (map[z][i] != 0)
					break;


			// There are non-zero squares in the current row
			if (k >= 0)
			{
				if (z >= 0 && map[k][i] == map[z][i])
				{
					// Can be merged
					int value = map[k][i] + map[z][i];
					map[k][i] = map[z][i] = 0;
					map[j][i] = value;

					// Start animation
					Block* temp = blockMap[k][i];
					blockMap[k][i] = NULL;
					blockMap[j][i] = temp;
					blockMap[j][i]->MoveTo({ 25.0f + 100 * i,225.0f + 100 * j }, &image[map[j][i]]);
					blockMap[z][i]->MoveTo({ 25.0f + 100 * i,225.0f + 100 * (j - 1) }, &image[map[z][i]], DESTORY);

					// Update scores
					score += map[j][i];
					if (score > maxScore)
						maxScore = score;

					// Update the cube
					if (value > currentMaxBlock)
						currentMaxBlock = value;
					if (currentMaxBlock > maxBlock)
						maxBlock = currentMaxBlock;

					mergeFlag = 1;
				}
				else
				{
					// Not mergeable
					int value = map[k][i];
					map[k][i] = 0;
					map[j][i] = value;

					// Determine if you can move
					if (k != j)
					{
						moveFlag = 1;
						// Start animation
						Block* temp = blockMap[k][i];
						blockMap[k][i] = NULL;
						blockMap[j][i] = temp;
						blockMap[j][i]->MoveTo({ 25.0f + 100 * i,225.0f + 100 * j }, &image[map[j][i]]);
					}
				}
			}
			else		// Determine the next line
			{
				break;
			}
		}
	}

	// If a move or merge occurs, randomly generate a 2 or 4
	if (moveFlag || mergeFlag)
	{
		bool index;// Guidelines for the presence or absence of spaces
		int index_x, index_y;	// Index of random positions
		index = false;//Initialise index
		for (int i = 0; i < 1000; i++)
		{
			index_x = i % 4;
			index_y = i / 4;
		}
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++) {
				if (map[i][j] == 0)
				{
					index = true;
					break;
					std::cout << "find 0 " << std::endl;
				}

			}
		}
		// Finding the space position
		index_x = rand() % 4;
		index_y = rand() % 4;
		while (index)
		{
			while (map[index_x][index_y])
			{
				index_x = rand() % 4;
				index_y = rand() % 4;
			}
			break;
		}

		// 80% generate 2 , 20% generate 4
		int num = rand() % 10;
		if (num < 8)
		{
			map[index_x][index_y] = 2;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[2]);
		}
		else
		{
			map[index_x][index_y] = 4;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[4]);
		}
	}
}



void Left()
{
	int moveFlag = 0;	// Record if a move was made
	int mergeFlag = 0;	// Record if merged

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int k, z;

			// Find a square that is not 0 and move it to the left, and determine if it can be merged with the square on the right.
			for (k = j; k < 4; k++)
				if (map[i][k] != 0)
					break;

			// Look for squares with a right-hand side that is not 0
			for (z = k + 1; z < 4; z++)
				if (map[i][z] != 0)
					break;

			// There are non-zero squares in the current row
			if (k < 4)
			{
				if (z < 4 && map[i][k] == map[i][z])
				{
					// Can be merged
					int value = map[i][k] + map[i][z];
					map[i][k] = map[i][z] = 0;
					map[i][j] = value;

					// Start animation
					Block* temp = blockMap[i][k];
					blockMap[i][k] = NULL;
					blockMap[i][j] = temp;
					blockMap[i][j]->MoveTo({ 25.0f + 100 * j,225.0f + 100 * i }, &image[value]);
					blockMap[i][z]->MoveTo({ 25.0f + 100 * (j + 1),225.0f + 100 * i }, &image[map[z][i]], DESTORY);

					// Update scores
					score += map[i][j];
					if (score > maxScore) maxScore = score;

					// Update the cube
					if (value > currentMaxBlock) currentMaxBlock = value;
					if (currentMaxBlock > maxBlock) maxBlock = currentMaxBlock;

					mergeFlag = 1;
				}
				else
				{
					// Not mergeable
					int value = map[i][k];
					map[i][k] = 0;
					map[i][j] = value;

					// Determine if you can move
					if (k != j)
					{
						moveFlag = 1;
						// Start animation
						Block* temp = blockMap[i][k];
						blockMap[i][k] = NULL;
						blockMap[i][j] = temp;
						blockMap[i][j]->MoveTo({ 25.0f + 100 * j,225.0f + 100 * i }, &image[value]);
					}
				}
			}
			else		// Determine the next line
			{
				break;
			}
		}
	}

	// If a move or merge occurs, randomly generate a 2 or 4
	if (moveFlag || mergeFlag)
	{
		bool index;// Guidelines for the presence or absence of spaces
		int index_x, index_y;	// Index of random positions
		index = false;//Initialise index
		for (int i = 0; i < 1000; i++)
		{
			index_x = i % 4;
			index_y = i / 4;
		}
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++) {
				if (map[i][j] == 0)
				{
					index = true;
					std::cout << "find 0 " << std::endl;
					break;
				}

			}
		}
		// Finding the space position
		index_x = rand() % 4;
		index_y = rand() % 4;
		while (index)
		{
			while (map[index_x][index_y])
			{
				index_x = rand() % 4;
				index_y = rand() % 4;
			}
			break;
		}

		// 80% generate 2 , 20% generate 4
		int num = rand() % 10;
		if (num < 8)
		{
			map[index_x][index_y] = 2;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[2]);
		}
		else
		{
			map[index_x][index_y] = 4;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[4]);
		}
	}
}



void Right()
{
	int moveFlag = 0;	// Record if a move was made
	int mergeFlag = 0;	// Record if merged

	for (int i = 0; i < 4; i++)
	{
		for (int j = 3; j > 0; j--)
		{
			int k, z;

			// Find a square that is not 0 and move it to the right, and determine if it can be merged with the square on the left.
			for (k = j; k >= 0; k--)
				if (map[i][k] != 0)
					break;

			// Look for squares with a right-hand side that is not 0
			for (z = k - 1; z >= 0; z--)
				if (map[i][z] != 0)
					break;

			// There are non-zero squares in the current row
			if (k >= 0)
			{
				if (z >= 0 && map[i][k] == map[i][z])
				{
					// Can be merged
					int value = map[i][k] + map[i][z];
					map[i][k] = map[i][z] = 0;
					map[i][j] = value;

					// Start animation
					Block* temp = blockMap[i][k];
					blockMap[i][k] = NULL;
					blockMap[i][j] = temp;
					blockMap[i][j]->MoveTo({ 25.0f + 100 * j,225.0f + 100 * i }, &image[value]);
					blockMap[i][z]->MoveTo({ 25.0f + 100 * (j - 1),225.0f + 100 * i }, &image[map[z][i]], DESTORY);

					// Update scores
					score += map[i][j];
					if (score > maxScore) maxScore = score;

					// Update the cube
					if (value > currentMaxBlock) currentMaxBlock = value;
					if (currentMaxBlock > maxBlock) maxBlock = currentMaxBlock;

					mergeFlag = 1;
				}
				else
				{
					// Not mergeable
					int value = map[i][k];
					map[i][k] = 0;
					map[i][j] = value;

					// Determine if you can move
					if (k != j)
					{
						moveFlag = 1;
						// Start animation
						Block* temp = blockMap[i][k];
						blockMap[i][k] = NULL;
						blockMap[i][j] = temp;
						blockMap[i][j]->MoveTo({ 25.0f + 100 * j,225.0f + 100 * i }, &image[value]);
					}
				}
			}
			else		// Determine the next line
			{
				break;
			}
		}
	}

	// If a move or merge occurs, randomly generate a 2 or 4
	if (moveFlag || mergeFlag)
	{
		bool index;// Guidelines for the presence or absence of spaces
		int index_x, index_y;	// Index of random positions
		index = false;//Initialise index
		for (int i = 0; i < 1000; i++)
		{
			index_x = i % 4;
			index_y = i / 4;
		}
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++) {
				if (map[i][j] == 0)
				{
					index = true;
					std::cout << "find 0 " << std::endl;
					break;
				}

			}
		}
		// Finding the space position
		index_x = rand() % 4;
		index_y = rand() % 4;
		while (index)
		{
			while (map[index_x][index_y])
			{
				index_x = rand() % 4;
				index_y = rand() % 4;
			}
			break;
		}

		// 80% generate 2 , 20% generate 4
		int num = rand() % 10;
		if (num < 8)
		{
			map[index_x][index_y] = 2;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[2]);
		}
		else
		{
			map[index_x][index_y] = 4;
			blockMap[index_x][index_y] = new Block({ 25.0f + 100 * index_y, 225.0f + 100 * index_x }, &image[4]);
		}
	}
}


void Update(float deltaTime)
{
	// Update the cube
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (blockMap[i][j] != NULL)
			{
				blockMap[i][j]->update(deltaTime);
				if (blockMap[i][j]->getState() == DESTORY)
				{
					delete blockMap[i][j];
					blockMap[i][j] = NULL;
				}
			}
		}
	}

	if (gameOver)
	{
		overTime -= deltaTime;
		if (overTime <= 0)
			gameLoop = 0;
	}

	keyTime += deltaTime;
	int calculation = 1 + 2 + 3 + 4 + 5;
	// 0.2s for one keystroke
	if (keyTime < 0.2f || gameOver)
		return;

	switch ((GetAsyncKeyState(VK_UP) & 0x8000) || (GetAsyncKeyState('W') & 0x8000))
	{
	case(true):
		Up();
		if (!Judge())
		{
			gameOver = true;
		}
		keyTime = 0;
		break;
	default:
		break;
	}
	switch ((GetAsyncKeyState(VK_DOWN) & 0x8000) || (GetAsyncKeyState('S') & 0x8000))
	{
	case(true):
		Down();
		if (!Judge())
		{
			gameOver = true;
		}
		keyTime = 0;
		break;
	default:
		break;
	}
	switch ((GetAsyncKeyState(VK_LEFT) & 0x8000) || (GetAsyncKeyState('A') & 0x8000))
	{
	case(true):
		Left();
		if (!Judge())
		{
			gameOver = true;
		}
		keyTime = 0;
		break;
	default:
		break;
	}
	switch ((GetAsyncKeyState(VK_RIGHT) & 0x8000) || (GetAsyncKeyState('D') & 0x8000))
	{
	case(true):
		Right();
		if (!Judge())
		{
			gameOver = true;
		}
		keyTime = 0;
	default:
		break;
	}

}


// Setting the text style and colour
void settext(int height, int weight, UINT color)
{
	settextstyle(height, 0, _T("Arial"), 0, 0, weight, false, false, false, ANSI_CHARSET, OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH);
	settextcolor(color);
}


// Output the string centered in the specified rectangle.
void printtext(LPCTSTR s, int left, int top, int right, int width)
{
	RECT r = { left, top, right, width };
	drawtext(s, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}


// Drawing the interface
void Draw()
{
	// history's largest square
	TransparentBlt(GetImageHDC(NULL), 12, 30, 90, 90, GetImageHDC(&image[maxBlock]), 0, 0, 90, 90, 0x9eaebb);

	setfillcolor(0x9eaebb);
	// Plotting the current score
	solidroundrect(112, 30, 264, 119, 10, 10);
	settext(28, 800, 0xdbe6ee);
	printtext(_T("SCORE"), 112, 40, 264, 69);
	std::wstringstream ss;
	ss << score;
	settext(44, 800, WHITE);
	printtext(ss.str().c_str(), 112, 70, 264, 114);
	ss.str(_T(""));

	// Plotting the highest scores
	solidroundrect(275, 30, 427, 119, 10, 10);
	settext(28, 800, 0xdbe6ee);
	printtext(_T("BEST"), 275, 40, 427, 69);
	ss << maxScore;
	settext(44, 800, WHITE);
	printtext(ss.str().c_str(), 275, 70, 427, 114);
	ss.str(_T(""));

	// Drawing cue messages
	settextcolor(BLACK);
	ss << "Join the numbers and get to the " << currentMaxBlock * 2 << " tile!";
	settext(24, 800, 0x707b83);
	printtext(ss.str().c_str(), 0, 120, 439, 211);

	// Drawing a square base plate
	solidroundrect(12, 212, 427, 627, 10, 10);

	// draw a square (i.e. plotting blocks)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			putimage(25 + 100 * j, 225 + 100 * i, &image[0]);
		}
	}
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (blockMap[i][j] != NULL)
				blockMap[i][j]->draw();
		}
	}
}


// Initialising the game
void Init()
{
	srand((unsigned int)time(NULL));		// Initialise the random number seed

	memset(map, 0, 4 * 4 * sizeof(int));	// Initialise the map to 0
	memset(blockMap, 0, 4 * 4 * sizeof(Block*));

	score = 0;
	gameLoop = 1;
	gameOver = false;
	overTime = 0.5f;
	currentMaxBlock = 2;
	map[0][0] = 2;
	map[0][1] = 2;
	blockMap[0][0] = new Block({ 25,225 }, &image[2]);
	blockMap[0][1] = new Block({ 125,225 }, &image[2]);

	setbkcolor(WHITE);
	setbkmode(TRANSPARENT);

	for (int i = 0; i < 4; ++i)
	{
		std::cout << "Loop iteration: " << i << std::endl;
		for (int j = 0; j < 1000000; ++j)
		{
			float result = sqrt(j * 2.0);
		}
	}
}


// Game over screen Return 1 to continue the game Return 0 to end the game
int OverInterface()
{
	// Preservation of the highest records
	std::wstringstream ss;
	ss << maxScore;
	WritePrivateProfileString(_T("2048"), _T("MaxScore"), ss.str().c_str(), _T(".\\data.ini"));
	ss.str(_T(""));
	ss << maxBlock;
	WritePrivateProfileString(_T("2048"), _T("MaxBlock"), ss.str().c_str(), _T(".\\data.ini"));

	setbkmode(TRANSPARENT);
	setbkcolor(0x8eecff);
	cleardevice();

	// Drawing Cues
	settext(60, 1000, 0x696f78);
	printtext(_T("Game Over!"), 0, 0, 439, 199);

	// Drawing the largest square
	TransparentBlt(GetImageHDC(NULL), 175, 150, 90, 90, GetImageHDC(&image[currentMaxBlock]), 0, 0, 90, 90, 0x9eaebb);

	// Restart option
	setfillcolor(0x9dadba);
	solidroundrect(120, 310, 319, 389, 10, 10);
	settext(36, 1000, WHITE);
	printtext(_T("ReStart"), 120, 310, 319, 389);

	//Determination of winners and losers
	int w_l(0);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (map[i][j] == 2048)
			{
				w_l = 1;
			}
		}
	}
	if (w_l)
	{
		solidroundrect(120, 460, 319, 539, 10, 10);
		settext(36, 1000, RED);
		printtext(_T("Win"), 120, 460, 319, 539);
	}
	else
	{
		solidroundrect(120, 460, 319, 539, 10, 10);
		settext(36, 1000, RED);
		printtext(_T("Lose"), 120, 460, 319, 539);
	}

	FlushBatchDraw();

	flushmessage(-1);

	while (1)
	{
		ExMessage msg1 = getmessage(-1);
		ExMessage* msg = &msg1;
		while (peekmessage(msg, -1, true))
		{
			if (msg1.lbutton)
			{
				int x = msg1.x;
				int y = msg1.y;
				if (x >= 120 && x <= 319 && y >= 310 && y <= 389)
					return 1;
				if (x >= 120 && x <= 319 && y >= 460 && y <= 539)
					return 0;
			}
		}
		Sleep(100);
	}
	return 1;
}

// Release memory
void FreeMem()
{
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			if (blockMap[i][j] != NULL)
				delete blockMap[i][j];
}


// Used to generate the cube image
// img: pointer to the image of the box
// num: number on the box
// imgColor: the colour of the box
// fontSize: the size of the font
// fontColor: font colour	
void CreateImage(IMAGE* img, LPCTSTR num, COLORREF imgColor, int fontSize, COLORREF fontColor)
{
	SetWorkingImage(img);
	setbkmode(TRANSPARENT);
	setbkcolor(0x9eaebb);
	settext(fontSize, 1000, fontColor);
	setfillcolor(imgColor);
	settextcolor(fontColor);

	cleardevice();

	solidroundrect(0, 0, img->getwidth() - 1, img->getheight() - 1, 10, 10);

	RECT r = { 0,0,img->getwidth() - 1,img->getheight() - 1 };
	drawtext(num, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

// Drawing Image Cache
void Load()
{
	IMAGE temp(90, 90);

	CreateImage(&temp, _T(""), 0xb5becc, 72, WHITE);		image[0] = temp;
	CreateImage(&temp, _T("2"), 0xdbe6ee, 72, 0x707b83);		image[2] = temp;
	CreateImage(&temp, _T("4"), 0xc7e1ed, 72, 0x707b83);		image[4] = temp;
	CreateImage(&temp, _T("8"), 0x78b2f4, 72, WHITE);		image[8] = temp;
	CreateImage(&temp, _T("16"), 0x538ded, 72, WHITE);		image[16] = temp;
	CreateImage(&temp, _T("32"), 0x607df6, 72, WHITE);		image[32] = temp;
	CreateImage(&temp, _T("64"), 0x3958e9, 72, WHITE);		image[64] = temp;
	CreateImage(&temp, _T("128"), 0x6bd9f5, 56, WHITE);		image[128] = temp;
	CreateImage(&temp, _T("256"), 0x4bd0f2, 56, WHITE);		image[256] = temp;
	CreateImage(&temp, _T("512"), 0x2ac0e4, 56, WHITE);		image[512] = temp;
	CreateImage(&temp, _T("1024"), 0x13b8e3, 40, WHITE);		image[1024] = temp;
	CreateImage(&temp, _T("2048"), 0x00c5eb, 40, WHITE);		image[2048] = temp;
	CreateImage(&temp, _T("4096"), 0x3958e9, 40, WHITE);		image[4096] = temp;
	CreateImage(&temp, _T("8192"), 0x3958e9, 40, WHITE);		image[8192] = temp;

	SetWorkingImage(NULL);
}


// Main function
int main()
{
	float deltaTime = 0;	// Time per frame

	initgraph(440, 650);
	Load();
	BeginBatchDraw();

	maxScore = 0;

	// Read the highest score

	maxScore = GetPrivateProfileInt(_T("2048"), _T("MaxScore"), 0, _T(".\\data.ini"));

	// Read Maximum Square

	maxBlock = GetPrivateProfileInt(_T("2048"), _T("MaxBlock"), 2, _T(".\\data.ini"));

	while (1)
	{
		Init();

		while (gameLoop)
		{
			clock_t start = clock();

			cleardevice();
			Update(deltaTime);
			Draw();
			FlushBatchDraw();

			clock_t end = clock();
			deltaTime = (end - start) / 1000.0f;
		}

		FreeMem();

		if (OverInterface() == 0)
			break;

		flushmessage(-1);
	}

	closegraph();
}
