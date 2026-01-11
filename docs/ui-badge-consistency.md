# UI Badge Consistency: Web vs iOS Dashboard

This document compares badge visibility and naming between the Web Dashboard and the iOS application UI.

## 1. Web Dashboard Reference
Source: https://fabric.openmind.org/

Visible badges:
- Mapper (View)
- Evaluator (Explore)
- Developer (Explore)
- Telenaut (Explore)
- Researcher (Explore)
- Personhood (View)
- Backpack (View)

Ranking Level:
- Emerald (League / Top 5%)

The Web Dashboard is treated as the reference implementation for badge naming and visibility.

## 2. iOS App Reference
Source: Screenshots and descriptions shared in Issue #1267.

Reported problems:
- Some badges missing in the iOS UI
- Naming mismatch between Web and iOS
- Inconsistent visibility rules (View vs Explore)

## 3. Observed Problems
- Missing badge rendering on iOS
- Badge name variations between platforms
- User confusion due to inconsistent UI behavior

## 4. Suggested Fix
- Use Web Dashboard badge names as the standard
- Ensure all Web-visible badges appear in iOS
- Keep consistent button actions (View / Explore)
- Centralize badge definitions in a shared config

## 5. Benefit
This improves:
- UI consistency
- User trust
- Developer maintenance
- Cross-platform clarity
