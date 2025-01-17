# ADR-003: Hybrid PDF Ingestion with RapidOCR (ONNX Runtime)

**Status:** Accepted  
**Date:** Day 07 — Scanned Page Detection & RapidOCR Fallback Integration  
**Author:** Prashant Bastola  
**Deciders:** Prashant Bastola

## Context

GroqFlow ingests user-uploaded PDF documents. Ingesting both selectable text PDFs and scanned document images without manual tagging is required.

## Problem

Full-document OCR on every page of every uploaded PDF creates 30+ second latencies and high memory usage, even when 95% of uploaded files are digital.

## Decision

Implement a **character density heuristic threshold** (40 characters). Only pages yielding fewer than 40 characters from `pypdf` are rendered via `PyMuPDF` and processed through `RapidOCR` with its lightweight ONNX Runtime backend.
