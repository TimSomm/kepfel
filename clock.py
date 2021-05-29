import cv2, math
import numpy as np

paths = [
  'input/clock.jpg',
  'input/clocknoise.jpg',
  'input/test1.png',
  'input/test2.jpg',
  'input/test3.jpg',
  'input/test4.jpg',
  'input/test5.jpg',
  'input/test6.jpg',
  'input/test7.jpg',
  'input/test8.png',
]

TESTNUM = len(paths)
# TESTNUM = 4
K = 2
WIDTH = 500
HEIGHT = 500
LINETRESH = 50
MINLINELENGTH = 20
MAXLINEGAP = 100

def preprocessImg(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  _, binaryImg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  iterations = 1
  binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_ERODE, struct, None, None, iterations, cv2.BORDER_REFLECT101)

  cv2.imshow('binary', cv2.resize(binaryImg, (WIDTH, HEIGHT), cv2.INTER_AREA))
  cv2.waitKey(0)
  return binaryImg

def findContoursOnBinary(binaryImg, possibleContours):
  contours, _ = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  for cntrIdx in range(0, len(contours)):
      # print(contours[cntrIdx].shape)
      if (contours[cntrIdx].shape[0] > binaryImg.shape[0]):
        possibleContours.append(contours[cntrIdx])

  return possibleContours

def findPossibleContours(binaryImg):
  possibleContours = findContoursOnBinary(binaryImg, [])

  struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  iterations = 1

  while possibleContours == []:
    binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_DILATE, struct, None, None, iterations, cv2.BORDER_REFLECT101)
    possibleContours = findContoursOnBinary(binaryImg, possibleContours)

  return possibleContours

def createSegmentImg(img, possibleContours):
  segment = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
  cv2.drawContours(segment, possibleContours, len(possibleContours) - 1, (255,0,0), -1, cv2.LINE_4)
  cv2.drawContours(img, possibleContours, len(possibleContours) - 1, (255,0,0), -1, cv2.LINE_4)
  cv2.imshow('img', cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA))
  cv2.imshow('segment', cv2.resize(segment, (WIDTH, HEIGHT), cv2.INTER_AREA))
  cv2.waitKey(0)
  return segment

def findAllLines(img, segment):
  lines = cv2.HoughLinesP(segment, 1, np.pi / 180, LINETRESH, None, MINLINELENGTH, MAXLINEGAP)

  X1 = []
  X2 = []
  Y1 = []
  Y2 = []

  for [currentLine] in lines:

    x1 = currentLine[0]
    y1 = currentLine[1]
    x2 = currentLine[2]
    y2 = currentLine[3]

    X1.append(x1)
    X2.append(x2)
    Y1.append(y1)
    Y2.append(y2)

    cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    line = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
  cv2.imshow('Lines', line)
  cv2.waitKey(0)
  return X1, X2, Y1, Y2

def createStackedLines(X1, X2, Y1, Y2):
  X1 = np.array(X1)
  Y1 = np.array(Y1)
  X2 = np.array(X2)
  Y2 = np.array(Y2)

  X1dash = X1.reshape(-1,1)
  Y1dash = Y1.reshape(-1,1)
  X2dash = X2.reshape(-1,1)
  Y2dash = Y2.reshape(-1,1)

  stacked = np.hstack((X1dash, Y1dash, X2dash, Y2dash))
  floatP = np.float32(stacked)
  return floatP

def createLinePointsWithKmeans(floatP, originalImg):
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  _, _, center = cv2.kmeans(floatP, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  finalImg = originalImg.copy()

  firstPts = []
  secondPts = []

  for p in range(len(center)):

    x1 = int(center[p][0])
    y1 = int(center[p][1])
    x2 = int(center[p][2])
    y2 = int(center[p][3])

    pt1 = (x1, y1)
    pt2 = (x2, y2)
    
    print(f'Line points: x = {pt1}, y = {pt2}')
    firstPts.append(pt1)
    secondPts.append(pt2)

    cv2.line(finalImg, pt1, pt2, (0, 255, 0), 2)
    resized = cv2.resize(finalImg, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
    cv2.imshow('Final',resized)

  cv2.waitKey(0)
  return firstPts, secondPts, finalImg

def createVectorsAndEquations(firstPts, secondPts):
  directionVectors = []
  normalVectors = []
  linesEq = []

  for (pt1, pt2) in zip(firstPts, secondPts):
    (x1, y1) = pt1
    (x2, y2) = pt2

    v1 = x2 - x1
    v2 = y2 - y1
    gcd = math.gcd(v1, v2)
    v1 = int(v1 / gcd)
    v2 = int(v2 / gcd)
    n1 = -v2
    n2 = v1
    c = int(n1 * x1 + n2 * y1)

    directionVectors.append([v1, v2])
    normalVectors.append([n1, n2])
    linesEq.append([n1, n2, c])
  
  return directionVectors, normalVectors, linesEq

def gaussElimination(linesEq):
  mul = linesEq[0][0]/linesEq[1][0] if linesEq[1][0] != 0 else 1
  for pts in linesEq[1:]:
    temp = [[y*mul for y in pts]]
    linesEq[1:] = temp

  return linesEq

def findIntersectionPoints(linesEq, finalImg):
  
  # a = 0, a1 - a2 (linesEq[0][0] - linesEq[1][0]), b = b1 - b2 (linesEq[0][1] - linesEq[1][1]), c = c1 - c2 (linesEq[0][2] - linesEq[1][2])
  # y = c / b
  # x = (c1 - y) / a1
  # i need only a1 to compute everything, so i call it a

  a = linesEq[0][0]

  b1 = linesEq[0][1]
  b2 = linesEq[1][1]
  b = b1 - b2

  c1 = linesEq[0][2]
  c2 = linesEq[1][2]
  c = c1 - c2


  intersectionY = int(c / b) if b != 0 else 0
  intersectionX = int((c1 - (b1*intersectionY))/a) if a != 0 else 0
  print(f'Intersection points: x = {intersectionX}, y = {intersectionY}')

  cv2.circle(finalImg, (intersectionX, intersectionY), 3, (255, 0, 0), -1)
  cv2.line(finalImg, (intersectionX, 0), (intersectionX, finalImg.shape[0]), (0, 255, 0), 1)
  circle = cv2.resize(finalImg, (WIDTH, HEIGHT), cv2.INTER_AREA)
  cv2.imshow('Circles', circle)

  return intersectionX, intersectionY

def getFurthestPoints(firstPts, secondPts, intersectionX, intersectionY):
  furtherPoints = []

  for (pt1, pt2) in zip(firstPts, secondPts):
    (x1, y1) = pt1
    (x2, y2) = pt2

    bigger = 'first' if abs(x1 - intersectionX) + abs(y1 - intersectionY) > abs(x2 - intersectionX) + abs(y2 - intersectionY) else 'second'
    if bigger == 'first':
      furtherPoints.append([x1, y1])
    else:
      furtherPoints.append([x2, y2])

  furthestPoint = furtherPoints[0] if abs(furtherPoints[0][0] - intersectionX) + abs(furtherPoints[0][1] - intersectionY) > abs(furtherPoints[1][0] - intersectionX) + abs(furtherPoints[1][1] - intersectionY) else furtherPoints[1]
  print(f'Further points: {furtherPoints}')
  print(f'Furthest point: {furthestPoint}')

  return furtherPoints, furthestPoint

def getAngles(directionVectors, furthestPoint, furtherPoints, intersectionX, intersectonY, img):
  minutePoint = furthestPoint;
  hourPoint = furtherPoints[0] if furthestPoint == furtherPoints[1] else furtherPoints[1]
  vectorMinute = directionVectors[0] if minutePoint == furtherPoints[0] else directionVectors[1]
  vectorHour = directionVectors[0] if vectorMinute == directionVectors[1] else directionVectors[1]
  intersectionVector = [0, img.shape[0]]

  unitVectorMinute = vectorMinute / np.linalg.norm(vectorMinute)
  unitVectorHour = vectorHour / np.linalg.norm(vectorHour)
  unitVectorIntersection = intersectionVector / np.linalg.norm(intersectionVector)
  dotProduct = np.dot(unitVectorIntersection, unitVectorMinute)


  angle = math.radians(180) - np.arccos(dotProduct) if furtherPoints[0][0] > intersectionX else (math.radians(180) - np.arccos(dotProduct)) + math.radians(180)
  minuteAngle = int(math.degrees(angle))
  print(f'Minute hand angle rounded: {minuteAngle}')
  dotProduct = np.dot(unitVectorHour, unitVectorIntersection)
  angle = math.radians(180) - np.arccos(dotProduct) if furtherPoints[1][0] > intersectionX else (math.radians(180) - np.arccos(dotProduct)) + math.radians(180)
  hourAngle = int(math.degrees(angle))
  print(f'Hour hand angle rounded: {hourAngle}')

  badUpperHalf = (minutePoint[1] > intersectonY and (minuteAngle < 90 or minuteAngle > 270)) or (hourPoint[1] > intersectonY and (hourAngle < 90 or hourAngle > 270))
  badBottomHalf = (minutePoint[1] < intersectonY and (minuteAngle > 90 and minuteAngle < 270)) or (hourPoint[1] < intersectonY and (hourAngle > 90 and hourAngle < 270))

  if badUpperHalf or badBottomHalf :
    minuteAngle = (minuteAngle + 180) % 360

  if badUpperHalf or badBottomHalf:
    hourAngle = (hourAngle + 180) % 360

  return minuteAngle, hourAngle

def calculateTime(minuteAngle, hourAngle):
  minutePartition = 360/60
  hourPartition = 360/12
  hour = round(hourAngle/hourPartition)
  minute = round(minuteAngle/minutePartition)
  additionalZeroHour = '' if len(str(hour)) > 1 else '0'
  additionalZeroMinute = '' if len(str(minute)) > 1 else '0'

  print(f'The time is: {additionalZeroHour}{hour}:{additionalZeroMinute}{minute}')

def whatIsTheTime(imagePath):
  img = cv2.imread(imagePath)
  originalImg = img.copy()
  cv2.imshow(imagePath, cv2.resize(originalImg, (WIDTH, HEIGHT), cv2.INTER_AREA))

  binaryImg = preprocessImg(img)
  possibleContours = findPossibleContours(binaryImg)
  segment = createSegmentImg(img, possibleContours)
  X1, X2, Y1, Y2 = findAllLines(img, segment)
  floatP = createStackedLines(X1, X2, Y1, Y2)
  firstPts, secondPts, finalImg = createLinePointsWithKmeans(floatP, originalImg)
  directionVectors, normalVectors, linesEq = createVectorsAndEquations(firstPts, secondPts)

  print(f'Direction vectors : {directionVectors}')
  print(f'Normal vectors : {normalVectors}')
  print(f'Equations: {linesEq}')
  linesEq = gaussElimination(linesEq)

  print(f'Equations after multiplying: {linesEq}')
  intersectionX, intersectionY = findIntersectionPoints(linesEq, finalImg)
  furtherPoints, furthestPoint = getFurthestPoints(firstPts, secondPts, intersectionX, intersectionY)
  minuteAngle, hourAngle = getAngles(directionVectors, furthestPoint, furtherPoints, intersectionX, intersectionY, img)
  calculateTime(minuteAngle, hourAngle)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

for i in range(TESTNUM):
  whatIsTheTime(paths[i])
