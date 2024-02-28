//============================================================================
//
// MM     MM  6666  555555  0000   2222
// MMMM MMMM 66  66 55     00  00 22  22
// MM MMM MM 66     55     00  00     22
// MM  M  MM 66666  55555  00  00  22222  --  "A 6502 Microprocessor Emulator"
// MM     MM 66  66     55 00  00 22
// MM     MM 66  66 55  55 00  00 22
// MM     MM  6666   5555   0000  222222
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: M6502Low.cxx,v 1.12 2007/01/01 18:04:51 stephena Exp $
//============================================================================

#include "M6502Low.hxx"
#include "Serializer.hxx"
#include "Deserializer.hxx"

#include <iostream>

namespace ale {
namespace stella {

#define debugStream std::cerr

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6502Low::M6502Low(uint32_t systemCyclesPerProcessorCycle)
    : M6502(systemCyclesPerProcessorCycle)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
M6502Low::~M6502Low()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
inline uint8_t M6502Low::peek(uint16_t address)
{
  uint8_t result = mySystem->peek(address);
  myLastAccessWasRead = true;
  return result;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
inline void M6502Low::poke(uint16_t address, uint8_t value)
{
  mySystem->poke(address, value);
  myLastAccessWasRead = false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool M6502Low::execute(uint32_t number)
{
  // Clear all of the execution status bits except for the fatal error bit
  myExecutionStatus &= FatalErrorBit;

  // Loop until execution is stopped or a fatal error occurs
  for(;;)
  {
    for(; !myExecutionStatus && (number != 0); --number)
    {
      uint16_t operandAddress = 0;
      uint8_t operand = 0;

#ifdef DEBUG
      debugStream << "PC=" << std::hex << setw(4) << PC << " ";
#endif

      // Fetch instruction at the program counter
      IR = peek(PC++);

#ifdef DEBUG
      debugStream << "IR=" << std::hex << setw(2) << (int)IR << " ";
      debugStream << "<" << ourAddressingModeTable[IR] << " ";
#endif

      // Update system cycles
      mySystem->incrementCycles(myInstructionSystemCycleTable[IR]);

      // Call code to execute the instruction
      switch(IR)
      {
        // 6502 instruction emulation is generated by an M4 macro file
        #include "M6502Low.ins"

        default:
          // Oops, illegal instruction executed so set fatal error flag
          myExecutionStatus |= FatalErrorBit;
          std::cerr << "Illegal Instruction! " << std::hex << (int) IR << std::endl;
      }

#ifdef DEBUG
      debugStream << std::hex << std::setw(4) << operandAddress << " ";
      debugStream << std::setw(4) << ourInstructionMnemonicTable[IR];
      debugStream << "> ";
      debugStream << "A=" << std::hex << std::setw(2) << (int)A << " ";
      debugStream << "X=" << std::hex << std::setw(2) << (int)X << " ";
      debugStream << "Y=" << std::hex << std::setw(2) << (int)Y << " ";
      debugStream << "PS=" << std::hex << std::setw(2) << (int)PS() << " ";
      debugStream << "SP=" << std::hex << std::setw(2) << (int)SP << " ";
      debugStream << "Cyc=" << std::dec << mySystem->cycles();
      debugStream << std::endl;
#endif
    }

    // See if we need to handle an interrupt
    if((myExecutionStatus & MaskableInterruptBit) ||
        (myExecutionStatus & NonmaskableInterruptBit))
    {
      // Yes, so handle the interrupt
      interruptHandler();
    }

    // See if execution has been stopped
    if(myExecutionStatus & StopExecutionBit)
    {
      // Yes, so answer that everything finished fine
      return true;
    }

    // See if a fatal error has occured
    if(myExecutionStatus & FatalErrorBit)
    {
      // Yes, so answer that something when wrong
      return false;
    }

    // See if we've executed the specified number of instructions
    if(number == 0)
    {
      // Yes, so answer that everything finished fine
      return true;
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void M6502Low::interruptHandler()
{
  // Handle the interrupt
  if((myExecutionStatus & MaskableInterruptBit) && !I)
  {
    mySystem->incrementCycles(7 * mySystemCyclesPerProcessorCycle);
    mySystem->poke(0x0100 + SP--, (PC - 1) >> 8);
    mySystem->poke(0x0100 + SP--, (PC - 1) & 0x00ff);
    mySystem->poke(0x0100 + SP--, PS() & (~0x10));
    D = false;
    I = true;
    PC = (uint16_t)mySystem->peek(0xFFFE) | ((uint16_t)mySystem->peek(0xFFFF) << 8);
  }
  else if(myExecutionStatus & NonmaskableInterruptBit)
  {
    mySystem->incrementCycles(7 * mySystemCyclesPerProcessorCycle);
    mySystem->poke(0x0100 + SP--, (PC - 1) >> 8);
    mySystem->poke(0x0100 + SP--, (PC - 1) & 0x00ff);
    mySystem->poke(0x0100 + SP--, PS() & (~0x10));
    D = false;
    PC = (uint16_t)mySystem->peek(0xFFFA) | ((uint16_t)mySystem->peek(0xFFFB) << 8);
  }

  // Clear the interrupt bits in myExecutionStatus
  myExecutionStatus &= ~(MaskableInterruptBit | NonmaskableInterruptBit);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool M6502Low::save(Serializer& out)
{
  std::string CPU = name();

  try
  {
    out.putString(CPU);

    out.putInt(A);    // Accumulator
    out.putInt(X);    // X index register
    out.putInt(Y);    // Y index register
    out.putInt(SP);   // Stack Pointer
    out.putInt(IR);   // Instruction register
    out.putInt(PC);   // Program Counter

    out.putBool(N);     // N flag for processor status register
    out.putBool(V);     // V flag for processor status register
    out.putBool(B);     // B flag for processor status register
    out.putBool(D);     // D flag for processor status register
    out.putBool(I);     // I flag for processor status register
    out.putBool(notZ);  // Z flag complement for processor status register
    out.putBool(C);     // C flag for processor status register

    out.putInt(myExecutionStatus);
  }
  catch(char *msg)
  {
    std::cerr << msg << std::endl;
    return false;
  }
  catch(...)
  {
    std::cerr << "Unknown error in save state for " << CPU << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool M6502Low::load(Deserializer& in)
{
  std::string CPU = name();

  try
  {
    if(in.getString() != CPU)
      return false;

    A = (uint8_t) in.getInt();    // Accumulator
    X = (uint8_t) in.getInt();    // X index register
    Y = (uint8_t) in.getInt();    // Y index register
    SP = (uint8_t) in.getInt();   // Stack Pointer
    IR = (uint8_t) in.getInt();   // Instruction register
    PC = (uint16_t) in.getInt();  // Program Counter

    N = in.getBool();     // N flag for processor status register
    V = in.getBool();     // V flag for processor status register
    B = in.getBool();     // B flag for processor status register
    D = in.getBool();     // D flag for processor status register
    I = in.getBool();     // I flag for processor status register
    notZ = in.getBool();  // Z flag complement for processor status register
    C = in.getBool();     // C flag for processor status register

    myExecutionStatus = (uint8_t) in.getInt();
  }
  catch(char *msg)
  {
    std::cerr << msg << std::endl;
    return false;
  }
  catch(...)
  {
    std::cerr << "Unknown error in load state for " << CPU << std::endl;
    return false;
  }

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const char* M6502Low::name() const
{
  return "M6502Low";
}

}  // namespace stella
}  // namespace ale
